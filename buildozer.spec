[app]
title = BlockBlastBot
package.name = bb_bot_pro
package.domain = org.ai.bb
source.dir = .
source.include_exts = py,png,jpg,kv,csv
version = 1.0

# Requisitos clave para tu Xeon y MCTS
requirements = python3, kivy, numpy, opencv-python, pyjnius, pandas

# Permisos CRÍTICOS para Xiaomi y autonomía
android.permissions = SYSTEM_ALERT_WINDOW, READ_EXTERNAL_STORAGE, FOREGROUND_SERVICE, FOREGROUND_SERVICE_MEDIA_PROJECTION
android.api = 30
android.minapi = 21

# Esto habilita el Servicio de Accesibilidad (las "manos" del bot)
android.services = bb_service:android_service.py

# IMPORTANTE: Esto añade la configuración de accesibilidad al manifiesto de Android
android.add_src = accessibility_service_config.xml
android.manifest.intent_filters = accessibility_service_config.xml
