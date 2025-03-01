FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Create conda environment
RUN conda create -n trading python=3.10 -y

# Install system dependencies 
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file instead of using Pipfile
COPY requirements.txt .

# Install dependencies using conda
SHELL ["/bin/bash", "-c"]
RUN conda init bash && \
    source ~/.bashrc && \
    conda activate trading && \
    pip install -r requirements.txt && \
    conda install pytorch -c pytorch -y

# Copy the .env file if it exists
COPY .env* ./

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p models logs data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH /opt/conda/envs/trading/bin:$PATH

# Command to run the application
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trading", "python"]
CMD ["train.py"]
