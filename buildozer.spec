[app]
title = BlockBlast Bot
package.name = blockblastbot
package.domain = org.seba

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0

requirements = python3,kivy,numpy,opencv-python-headless,pyjnius

orientation = portrait

osx.kivy_version = 2.3.0

android.permissions = SYSTEM_ALERT_WINDOW,FOREGROUND_SERVICE,FOREGROUND_SERVICE_MEDIA_PROJECTION,BIND_ACCESSIBILITY_SERVICE
android.archs = arm64-v8a
android.api = 33
android.sdk = 33
android.build_tools_version = 33.0.0
android.ndk = 25b
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
