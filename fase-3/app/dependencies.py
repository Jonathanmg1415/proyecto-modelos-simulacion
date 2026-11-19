from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings

# Alias para declarar la configuración como parámetro de un endpoint:

SettingsDep = Annotated[Settings, Depends(get_settings)]