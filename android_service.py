import time
import gc
import threading
import numpy as np

from jnius import autoclass, cast, PythonJavaClass, java_method
from block_blast_simulator import BlockBlastSim
from mcts_brain import BlockBlastBrain

# --- Android Autoclasses (Pyjnius) ---
Context = autoclass('android.content.Context')
PythonActivity = autoclass('org.kivy.android.PythonActivity')
ImageReader = autoclass('android.media.ImageReader')
PixelFormat = autoclass('android.graphics.PixelFormat')
DisplayManager = autoclass('android.hardware.display.DisplayManager')
Path = autoclass('android.graphics.Path')
GestureDescription = autoclass('android.accessibilityservice.GestureDescription')
GestureDescriptionBuilder = autoclass('android.accessibilityservice.GestureDescription$Builder')
StrokeDescription = autoclass('android.accessibilityservice.GestureDescription$StrokeDescription')
WindowManager = autoclass('android.view.WindowManager')
LayoutParams = autoclass('android.view.WindowManager$LayoutParams')
Gravity = autoclass('android.view.Gravity')
Button = autoclass('android.widget.Button')
Color = autoclass('android.graphics.Color')


# ==========================================
# 1. GESTOS DE ACCESIBILIDAD (Reemplazo ADB)
# ==========================================
def perform_android_swipe(accessibility_service, x1, y1, x2, y2, duration_ms=250):
    """
    Ejecuta un swipe genuino en Android simulando un dedo a través del AccessibilityService.
    Reemplaza completamente a PyAutoGUI o transacciones ADB que merman la velocidad.
    """
    path = Path()
    path.moveTo(float(x1), float(y1))
    path.lineTo(float(x2), float(y2))
    
    # Comienza en ms 0 y tarda duration_ms
    stroke = StrokeDescription(path, 0, duration_ms)
    builder = GestureDescriptionBuilder()
    builder.addStroke(stroke)
    gesture = builder.build()
    
    # Despachar al OS (sin callbacks)
    accessibility_service.dispatchGesture(gesture, None, None)


# ==========================================
# 2. CAPTURA NATIVA CON MEDIA PROJECTION
# ==========================================
class NativeScreenCapturer:
    def __init__(self, media_projection, width=1080, height=2340, density=440):
        # Medidas nativas, ajustables a la resolución final del Xiaomi
        self.width = width
        self.height = height
        
        # MaxImages = 2 (Crítico para que no asfixie la memoria de video)
        self.image_reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        
        # Casteo de pantalla al ImageReader en directo
        self.virtual_display = media_projection.createVirtualDisplay(
            "BlockBlastBot",
            width, height, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            self.image_reader.getSurface(),
            None, None
        )
        
    def get_latest_frame(self):
        image = self.image_reader.acquireLatestImage()
        if image is None:
            return None
            
        planes = image.getPlanes()
        buffer = planes[0].getBuffer()
        
        # Transmisión directa de Java Bytes -> Python Bytearray
        buffer_size = buffer.remaining()
        byte_array = bytearray(buffer_size)
        buffer.get(byte_array)
        
        # ==========================================
        # OPTIMIZACIÓN DE MEMORIA NATIVA (CRITICO XIAOMI)
        # ==========================================
        # Liberar la imagen Android INMEDIATAMENTE.
        # Si no llamas a close(), el sistema Android congelará la aplicación.
        image.close() 
        
        row_stride = planes[0].getRowStride()
        pixel_stride = planes[0].getPixelStride()
        
        arr = np.frombuffer(byte_array, dtype=np.uint8)
        
        # Se elimina la basura del row_stride nativo (Padding inter-pixel)
        arr = arr.reshape((self.height, row_stride // pixel_stride, pixel_stride))
        frame = arr[:, :self.width, :]
        
        return frame


# ==========================================
# 3. INTERFAZ FLOTANTE (KIVY-OVERLAY)
# ==========================================
class ClickListener(PythonJavaClass):
    # Implementa una interfaz de Java de forma nativa en Android/Kivy
    __javainterfaces__ = ['android/view/View$OnClickListener']

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    @java_method('(Landroid/view/View;)V')
    def onClick(self, view):
        self.callback(view)

class FloatingOverlay:
    def __init__(self, context, start_callback, stop_callback):
        self.context = context
        self.window_manager = cast('android.view.WindowManager', context.getSystemService(Context.WINDOW_SERVICE))
        self.running = False
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        
        # Params TYPE_APPLICATION_OVERLAY requieren del permiso AndroidManifest -> SYSTEM_ALERT_WINDOW
        params = LayoutParams(
            300, 150, # Size
            LayoutParams.TYPE_APPLICATION_OVERLAY, 
            LayoutParams.FLAG_NOT_FOCUSABLE, # Para no interrumpir el teclado si el usuario escribe
            PixelFormat.TRANSLUCENT
        )
        params.gravity = Gravity.TOP | Gravity.LEFT
        params.x = 20
        params.y = 200
        
        self.button = Button(context)
        self.button.setText("INICIAR BOT")
        self.button.setBackgroundColor(Color.parseColor("#A000FF00")) # Verde Translúcido
        self.button.setOnClickListener(ClickListener(self.on_click))
        
        self.window_manager.addView(self.button, params)

    def on_click(self, view):
        self.running = not self.running
        if self.running:
            self.button.setText("DETENER BOT")
            self.button.setBackgroundColor(Color.parseColor("#A0FF0000")) # Rojo transparente
            self.start_callback()
        else:
            self.button.setText("INICIAR BOT")
            self.button.setBackgroundColor(Color.parseColor("#A000FF00"))
            self.stop_callback()


# ==========================================
# 4. SERVICIO WORKER DEL BOT
# ==========================================
class BotServiceBackground:
    def __init__(self, accessibility_service, media_projection, context):
        self.accessibility_service = accessibility_service
        self.capturer = NativeScreenCapturer(media_projection)
        
        self.sim = BlockBlastSim()
        self.brain = BlockBlastBrain(self.sim)
        
        self.bot_active = False
        self.thread = None
        
        # Botón Flotante maestro
        self.overlay = FloatingOverlay(context, self.start_bot, self.stop_bot)

    def start_bot(self):
        self.bot_active = True
        self.thread = threading.Thread(target=self._bot_loop, daemon=True)
        self.thread.start()

    def stop_bot(self):
        self.bot_active = False
        if self.thread:
            self.thread.join()

    def _bot_loop(self):
        while self.bot_active:
            # 1. Adquisición NATIVA sin socket/ADB
            frame = self.capturer.get_latest_frame()
            
            if frame is not None:
                # --- INTEGRACIÓN A TU CV ---
                # Aca deberás mandar frame (numpy uint8) a tus detectores Canny/Cálculos de matriz
                # Ejemplo imaginario: 
                # board_actual, pieces = tu_codigo_vision(frame)
                
                # --- MCTS CEREBRO ---
                # self.sim.board = board_actual
                # self.sim.current_piece_indices = pieces
                pass  # <- best_moves = self.brain.think_and_play(time_limit_ms=100)
                
                # --- EJECUCIÓN DEL GESTO ---
                # perform_android_swipe(self.accessibility_service, ...)
                
                # ==========================================
                # OPTIMIZACIÓN DE MEMORIA PRINCIPAL (GC)
                # ==========================================
                # Cada frame crudo son como ~10 MB de RAM.  
                # Destruimos el frame a nivel de Python y purgamos.
                del frame
                gc.collect()

            # Evita estrangular la CPU en bucles vacíos de frames faltantes
            time.sleep(0.01)

def run_service(accessibility_service_instance, media_projection_instance):
    """
    Hook de inicialización. Buildozer o el framework de arranque de Python 
    llamará a esta función pasándole la UI y los adaptadores compilados.
    """
    context = cast('android.content.Context', PythonActivity.mActivity)
    bot = BotServiceBackground(accessibility_service_instance, media_projection_instance, context)
    return bot
