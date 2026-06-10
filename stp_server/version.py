"""Product identity for the STP registration `server` field.

PRODUCT_NAME is fixed by this implementation and is never user-configurable —
user-facing naming belongs in the provider's configured display name. The
`Name/Version` pair is the plugin's product identity on the wire (STP
`provider.register` `server` field) and may be shared by hosts for telemetry.
"""

PRODUCT_NAME = "ComfyUI-Stimma"
PRODUCT_VERSION = "1.1.0"
