{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "asyncio.run() cannot be called from a running event loop",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[1;32md:\\Programming\\Codes\\ProjectTechPark\\last.ipynb Cell 1\u001b[0m line \u001b[0;36m4\n\u001b[0;32m     <a href='vscode-notebook-cell:/d%3A/Programming/Codes/ProjectTechPark/last.ipynb#W1sZmlsZQ%3D%3D?line=38'>39</a>\u001b[0m \u001b[39mif\u001b[39;00m \u001b[39m__name__\u001b[39m \u001b[39m==\u001b[39m \u001b[39m\"\u001b[39m\u001b[39m__main__\u001b[39m\u001b[39m\"\u001b[39m:\n\u001b[0;32m     <a href='vscode-notebook-cell:/d%3A/Programming/Codes/ProjectTechPark/last.ipynb#W1sZmlsZQ%3D%3D?line=39'>40</a>\u001b[0m     \u001b[39mimport\u001b[39;00m \u001b[39muvicorn\u001b[39;00m\n\u001b[1;32m---> <a href='vscode-notebook-cell:/d%3A/Programming/Codes/ProjectTechPark/last.ipynb#W1sZmlsZQ%3D%3D?line=40'>41</a>\u001b[0m     uvicorn\u001b[39m.\u001b[39;49mrun(app, host\u001b[39m=\u001b[39;49m\u001b[39m\"\u001b[39;49m\u001b[39m0.0.0.0\u001b[39;49m\u001b[39m\"\u001b[39;49m, port\u001b[39m=\u001b[39;49m\u001b[39m8000\u001b[39;49m)\n",
      "File \u001b[1;32md:\\Programming\\Python\\Lib\\site-packages\\uvicorn\\main.py:587\u001b[0m, in \u001b[0;36mrun\u001b[1;34m(app, host, port, uds, fd, loop, http, ws, ws_max_size, ws_max_queue, ws_ping_interval, ws_ping_timeout, ws_per_message_deflate, lifespan, interface, reload, reload_dirs, reload_includes, reload_excludes, reload_delay, workers, env_file, log_config, log_level, access_log, proxy_headers, server_header, date_header, forwarded_allow_ips, root_path, limit_concurrency, backlog, limit_max_requests, timeout_keep_alive, timeout_graceful_shutdown, ssl_keyfile, ssl_certfile, ssl_keyfile_password, ssl_version, ssl_cert_reqs, ssl_ca_certs, ssl_ciphers, headers, use_colors, app_dir, factory, h11_max_incomplete_event_size)\u001b[0m\n\u001b[0;32m    585\u001b[0m     Multiprocess(config, target\u001b[39m=\u001b[39mserver\u001b[39m.\u001b[39mrun, sockets\u001b[39m=\u001b[39m[sock])\u001b[39m.\u001b[39mrun()\n\u001b[0;32m    586\u001b[0m \u001b[39melse\u001b[39;00m:\n\u001b[1;32m--> 587\u001b[0m     server\u001b[39m.\u001b[39;49mrun()\n\u001b[0;32m    588\u001b[0m \u001b[39mif\u001b[39;00m config\u001b[39m.\u001b[39muds \u001b[39mand\u001b[39;00m os\u001b[39m.\u001b[39mpath\u001b[39m.\u001b[39mexists(config\u001b[39m.\u001b[39muds):\n\u001b[0;32m    589\u001b[0m     os\u001b[39m.\u001b[39mremove(config\u001b[39m.\u001b[39muds)  \u001b[39m# pragma: py-win32\u001b[39;00m\n",
      "File \u001b[1;32md:\\Programming\\Python\\Lib\\site-packages\\uvicorn\\server.py:61\u001b[0m, in \u001b[0;36mServer.run\u001b[1;34m(self, sockets)\u001b[0m\n\u001b[0;32m     59\u001b[0m \u001b[39mdef\u001b[39;00m \u001b[39mrun\u001b[39m(\u001b[39mself\u001b[39m, sockets: Optional[List[socket\u001b[39m.\u001b[39msocket]] \u001b[39m=\u001b[39m \u001b[39mNone\u001b[39;00m) \u001b[39m-\u001b[39m\u001b[39m>\u001b[39m \u001b[39mNone\u001b[39;00m:\n\u001b[0;32m     60\u001b[0m     \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mconfig\u001b[39m.\u001b[39msetup_event_loop()\n\u001b[1;32m---> 61\u001b[0m     \u001b[39mreturn\u001b[39;00m asyncio\u001b[39m.\u001b[39;49mrun(\u001b[39mself\u001b[39;49m\u001b[39m.\u001b[39;49mserve(sockets\u001b[39m=\u001b[39;49msockets))\n",
      "File \u001b[1;32md:\\Programming\\Python\\Lib\\asyncio\\runners.py:186\u001b[0m, in \u001b[0;36mrun\u001b[1;34m(main, debug)\u001b[0m\n\u001b[0;32m    161\u001b[0m \u001b[39m\u001b[39m\u001b[39m\"\"\"Execute the coroutine and return the result.\u001b[39;00m\n\u001b[0;32m    162\u001b[0m \n\u001b[0;32m    163\u001b[0m \u001b[39mThis function runs the passed coroutine, taking care of\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    182\u001b[0m \u001b[39m    asyncio.run(main())\u001b[39;00m\n\u001b[0;32m    183\u001b[0m \u001b[39m\"\"\"\u001b[39;00m\n\u001b[0;32m    184\u001b[0m \u001b[39mif\u001b[39;00m events\u001b[39m.\u001b[39m_get_running_loop() \u001b[39mis\u001b[39;00m \u001b[39mnot\u001b[39;00m \u001b[39mNone\u001b[39;00m:\n\u001b[0;32m    185\u001b[0m     \u001b[39m# fail fast with short traceback\u001b[39;00m\n\u001b[1;32m--> 186\u001b[0m     \u001b[39mraise\u001b[39;00m \u001b[39mRuntimeError\u001b[39;00m(\n\u001b[0;32m    187\u001b[0m         \u001b[39m\"\u001b[39m\u001b[39masyncio.run() cannot be called from a running event loop\u001b[39m\u001b[39m\"\u001b[39m)\n\u001b[0;32m    189\u001b[0m \u001b[39mwith\u001b[39;00m Runner(debug\u001b[39m=\u001b[39mdebug) \u001b[39mas\u001b[39;00m runner:\n\u001b[0;32m    190\u001b[0m     \u001b[39mreturn\u001b[39;00m runner\u001b[39m.\u001b[39mrun(main)\n",
      "\u001b[1;31mRuntimeError\u001b[0m: asyncio.run() cannot be called from a running event loop"
     ]
    }
   ],
   "source": [
    "from fastapi import FastAPI, File, UploadFile\n",
    "from fastapi.responses import HTMLResponse\n",
    "from fastapi.templating import Jinja2Templates\n",
    "import cv2\n",
    "import io\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "from keras.models import load_model\n",
    "\n",
    "app = FastAPI()\n",
    "templates = Jinja2Templates(directory=\"templates\")\n",
    "\n",
    "# Load the pre-trained model\n",
    "model = load_model(\"my_model.keras\")\n",
    "\n",
    "# Define labels for predictions\n",
    "labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']\n",
    "\n",
    "# Function to predict the class of an uploaded image\n",
    "def predict_image(upload_file):\n",
    "    content = upload_file.file.read()\n",
    "    image = Image.open(io.BytesIO(content))\n",
    "    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)\n",
    "    img = cv2.resize(opencv_image, (150, 150))\n",
    "    img = img.reshape(1, 150, 150, 3)\n",
    "    predictions = model.predict(img)\n",
    "    predicted_class = labels[np.argmax(predictions)]\n",
    "    return predicted_class\n",
    "\n",
    "@app.post(\"/uploadfile/\")\n",
    "async def upload_file(file: UploadFile):\n",
    "    predicted_class = predict_image(file)\n",
    "    return {\"predicted_class\": predicted_class}\n",
    "\n",
    "@app.get(\"/\", response_class=HTMLResponse)\n",
    "async def read_item():\n",
    "    return templates.TemplateResponse(\"index.html\", {\"request\": None})\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    import uvicorn\n",
    "    uvicorn.run(app, host=\"0.0.0.0\", port=8000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
