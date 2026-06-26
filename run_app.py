import os

import uvicorn


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8510"))
    uvicorn.run("core.app_new:app", host="127.0.0.1", port=port, reload=False)
