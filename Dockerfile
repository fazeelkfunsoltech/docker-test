FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir runpod
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]