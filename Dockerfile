# Mayar Ariss, 2/10/2025
# MIT Senseable City Lab

# Use RAPIDS base image with CUDA 12.5 and Python 3.12
FROM rapidsai/base:24.12-cuda12.5-py3.12

# Set the working directory inside the container
WORKDIR /workspace

# Install required Python packages (REMOVE `logging`)
RUN pip install --no-cache-dir contextily folium geopandas hyperopt kneed matplotlib numpy pandas plotly scipy shapely seaborn tqdm notebook openpyxl optuna xgboost

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
