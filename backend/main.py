import asyncio
import os
import sys
import subprocess
import threading
import time
import json
import signal
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add the parent directory to the Python path to import trading bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Trading Bot API",
    description="Advanced RL Trading Bot API for training and inference",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000","http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global process tracking
training_process: Optional[subprocess.Popen] = None
inference_process: Optional[subprocess.Popen] = None
training_active = False
inference_active = False

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Pydantic models
class TrainingConfig(BaseModel):
    train_data: str
    symbol: str
    interval: Optional[str] = "1h"
    window_size: Optional[int] = 50
    leverage: Optional[float] = 1.0
    max_position: Optional[float] = 0.1
    balance: Optional[float] = 1000.0
    risk_reward: Optional[float] = 2.0
    stop_loss: Optional[float] = 0.02
    trade_fee: Optional[float] = 0.001
    architecture: Optional[str] = "transformer"
    embedding_dim: Optional[int] = 128
    n_encoder_layers: Optional[int] = 4
    n_decoder_layers: Optional[int] = 4
    dropout: Optional[float] = 0.1
    attention_type: Optional[str] = "standard"
    n_heads: Optional[int] = 8
    n_latents: Optional[int] = 32
    n_groups: Optional[int] = 4
    ffn_type: Optional[str] = "standard"
    ffn_dim: Optional[int] = 512
    n_experts: Optional[int] = 8
    top_k: Optional[int] = 2
    norm_type: Optional[str] = "layer"
    feature_extractor_type: Optional[str] = "cnn"
    feature_extractor_dim: Optional[int] = 64
    feature_extractor_layers: Optional[int] = 3
    head_hidden_dim: Optional[int] = 256
    head_n_layers: Optional[int] = 2
    lr: Optional[float] = 0.001
    gamma: Optional[float] = 0.99
    gae_lambda: Optional[float] = 0.95
    policy_clip: Optional[float] = 0.2
    batch_size: Optional[int] = 64
    n_epochs: Optional[int] = 10
    entropy_coef: Optional[float] = 0.01
    value_coef: Optional[float] = 0.5
    max_grad_norm: Optional[float] = 0.5
    weight_decay: Optional[float] = 1e-4
    episodes: Optional[int] = 1000
    n_splits: Optional[int] = 5
    val_ratio: Optional[float] = 0.2
    eval_freq: Optional[int] = 100
    save_path: Optional[str] = "best_model.pt"
    device: Optional[str] = "auto"
    dynamic_leverage: Optional[bool] = True
    use_risk_adjusted_rewards: Optional[bool] = True
    use_skip_connections: Optional[bool] = False
    use_layer_norm: Optional[bool] = False
    use_instance_norm: Optional[bool] = False
    head_use_layer_norm: Optional[bool] = False
    head_use_residual: Optional[bool] = False
    use_gae: Optional[bool] = True
    normalize_advantage: Optional[bool] = True

class InferenceConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_path: str
    scaler_path: str
    symbol: Optional[str] = "BTCUSDT"
    interval: Optional[str] = "1h"
    window_size: Optional[int] = 50
    leverage: Optional[float] = 1.0
    risk_reward_ratio: Optional[float] = 2.0
    stop_loss_percent: Optional[float] = 0.02
    initial_balance: Optional[float] = 1000.0
    base_url: Optional[str] = "https://testnet.binancefuture.com"
    sleep_time: Optional[int] = 60
    device: Optional[str] = "auto"
    exploration_rate: Optional[float] = 0.0
    allow_scaling: Optional[bool] = False
    dry_run: Optional[bool] = True

class FileList(BaseModel):
    models: List[str]
    data: List[str]
    scalers: List[str]

class DataConfig(BaseModel):
    symbol: str
    interval: Optional[str] = "1m"
    days: Optional[int] = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    output_dir: Optional[str] = "data"
    split: Optional[bool] = False
    test_ratio: Optional[float] = 0.2
    validation_ratio: Optional[float] = 0.0

class StatusResponse(BaseModel):
    training_active: bool
    inference_active: bool

# API Routes
@app.get("/", response_model=dict)
async def root():
    return {"message": "Trading Bot FastAPI Server", "status": "running", "version": "2.0.0"}

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        training_active=training_active,
        inference_active=inference_active
    )

@app.post("/api/process_data")
async def process_data(config: DataConfig):
    print(config)
    try:
        cmd = ['python3', 'data.py']
        cmd.extend(['--symbol', config.symbol])
        cmd.extend(['--interval', config.interval])
        cmd.extend(['--days', str(config.days)])
        if config.start_date:
            cmd.extend(['--start_date', config.start_date])
        if config.end_date:
            cmd.extend(['--end_date', config.end_date])
        if config.output_dir:
            cmd.extend(['--output_dir', config.output_dir])
        if config.split:
            cmd.append('--split')
            cmd.extend(['--test_ratio', str(config.test_ratio)])
            cmd.extend(['--validation_ratio', str(config.validation_ratio)])
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        # Build file path to returned CSV
        file_name = f"{config.symbol}_{config.interval}_with_metrics.csv"
        file_path = os.path.join(config.output_dir, file_name)
        return JSONResponse(content={
            "message": "Data processed successfully",
            "output": result.stdout,
            "file_path": file_path
        })
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start_training")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    global training_process, training_active
    
    if training_active:
        raise HTTPException(status_code=400, detail="Training is already running")
    
    try:
        # Build training command
        cmd = [sys.executable, '-u', 'train.py']
        
        # Add required parameters
        cmd.extend(['--train_data', config.train_data])
        cmd.extend(['--symbol', config.symbol])
        
        # Add optional parameters
        optional_params = {
            'interval': config.interval,
            'window': config.window_size,
            'leverage': config.leverage,
            'max_position': config.max_position,
            'balance': config.balance,
            'risk_reward': config.risk_reward,
            'stop_loss': config.stop_loss,
            'trade_fee': config.trade_fee,
            'architecture': config.architecture,
            'embedding_dim': config.embedding_dim,
            'n_encoder_layers': config.n_encoder_layers,
            'n_decoder_layers': config.n_decoder_layers,
            'dropout': config.dropout,
            'attention_type': config.attention_type,
            'n_heads': config.n_heads,
            'n_latents': config.n_latents,
            'n_groups': config.n_groups,
            'ffn_type': config.ffn_type,
            'ffn_dim': config.ffn_dim,
            'n_experts': config.n_experts,
            'top_k': config.top_k,
            'norm_type': config.norm_type,
            'feature_extractor_type': config.feature_extractor_type,
            'feature_extractor_dim': config.feature_extractor_dim,
            'feature_extractor_layers': config.feature_extractor_layers,
            'head_hidden_dim': config.head_hidden_dim,
            'head_n_layers': config.head_n_layers,
            'lr': config.lr,
            'gamma': config.gamma,
            'gae_lambda': config.gae_lambda,
            'policy_clip': config.policy_clip,
            'batch_size': config.batch_size,
            'n_epochs': config.n_epochs,
            'entropy_coef': config.entropy_coef,
            'value_coef': config.value_coef,
            'max_grad_norm': config.max_grad_norm,
            'weight_decay': config.weight_decay,
            'episodes': config.episodes,
            'n_splits': config.n_splits,
            'val_ratio': config.val_ratio,
            'eval_freq': config.eval_freq,
            'save_path': config.save_path,
            'device': config.device
        }
        
        # Map parameter names to command line flags
        param_flags = {
            'interval': '--interval',
            'window': '--window',
            'leverage': '--leverage',
            'max_position': '--max_position',
            'balance': '--balance',
            'risk_reward': '--risk_reward',
            'stop_loss': '--stop_loss',
            'trade_fee': '--trade_fee',
            'architecture': '--architecture',
            'embedding_dim': '--embedding_dim',
            'n_encoder_layers': '--n_encoder_layers',
            'n_decoder_layers': '--n_decoder_layers',
            'dropout': '--dropout',
            'attention_type': '--attention_type',
            'n_heads': '--n_heads',
            'n_latents': '--n_latents',
            'n_groups': '--n_groups',
            'ffn_type': '--ffn_type',
            'ffn_dim': '--ffn_dim',
            'n_experts': '--n_experts',
            'top_k': '--top_k',
            'norm_type': '--norm_type',
            'feature_extractor_type': '--feature_extractor_type',
            'feature_extractor_dim': '--feature_extractor_dim',
            'feature_extractor_layers': '--feature_extractor_layers',
            'head_hidden_dim': '--head_hidden_dim',
            'head_n_layers': '--head_n_layers',
            'lr': '--lr',
            'gamma': '--gamma',
            'gae_lambda': '--gae_lambda',
            'policy_clip': '--policy_clip',
            'batch_size': '--batch_size',
            'n_epochs': '--n_epochs',
            'entropy_coef': '--entropy_coef',
            'value_coef': '--value_coef',
            'max_grad_norm': '--max_grad_norm',
            'weight_decay': '--weight_decay',
            'episodes': '--episodes',
            'n_splits': '--n_splits',
            'val_ratio': '--val_ratio',
            'eval_freq': '--eval_freq',
            'save_path': '--save_path',
            'device': '--device'
        }
        
        for param, value in optional_params.items():
            if value is not None and param in param_flags:
                cmd.extend([param_flags[param], str(value)])
        
        # Handle boolean flags
        if not config.dynamic_leverage:
            cmd.append('--static_leverage')
        if not config.use_risk_adjusted_rewards:
            cmd.append('--simple_rewards')
        if config.use_skip_connections:
            cmd.append('--use_skip_connections')
        if config.use_layer_norm:
            cmd.append('--use_layer_norm')
        if config.use_instance_norm:
            cmd.append('--use_instance_norm')
        if config.head_use_layer_norm:
            cmd.append('--head_use_layer_norm')
        if config.head_use_residual:
            cmd.append('--head_use_residual')
        if config.use_gae:
            cmd.append('--use_gae')
        if config.normalize_advantage:
            cmd.append('--normalize_advantage')
        
        # Start training process
        logger.info(f"Executing training command: {' '.join(cmd)}")
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        training_active = True
        
        # Start background task to monitor training output
        background_tasks.add_task(monitor_training_output)
        
        return {"message": "Training started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop_training")
async def stop_training():
    global training_process, training_active
    
    if not training_active or not training_process:
        raise HTTPException(status_code=400, detail="No training process is running")
    
    try:
        # Terminate the process
        training_process.terminate()
        try:
            training_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            training_process.kill()
        
        training_active = False
        
        await manager.broadcast({
            "type": "training_output",
            "data": "Training stopped by user\n",
            "output_type": "info",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"message": "Training stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start_inference")
async def start_inference(config: InferenceConfig, background_tasks: BackgroundTasks):
    global inference_process, inference_active
    
    if inference_active:
        raise HTTPException(status_code=400, detail="Inference is already running")
    
    try:
        # Build inference command
        cmd = ['python', 'inference.py']
        
        # Add required parameters
        cmd.extend(['--model_path', config.model_path])
        cmd.extend(['--scaler_path', config.scaler_path])
        
        # Add optional parameters
        optional_params = {
            'symbol': config.symbol,
            'interval': config.interval,
            'window_size': config.window_size,
            'leverage': config.leverage,
            'risk_reward_ratio': config.risk_reward_ratio,
            'stop_loss_percent': config.stop_loss_percent,
            'initial_balance': config.initial_balance,
            'base_url': config.base_url,
            'sleep_time': config.sleep_time,
            'device': config.device,
            'exploration_rate': config.exploration_rate
        }
        
        param_flags = {
            'symbol': '--symbol',
            'interval': '--interval',
            'window_size': '--window_size',
            'leverage': '--leverage',
            'risk_reward_ratio': '--risk_reward_ratio',
            'stop_loss_percent': '--stop_loss_percent',
            'initial_balance': '--initial_balance',
            'base_url': '--base_url',
            'sleep_time': '--sleep_time',
            'device': '--device',
            'exploration_rate': '--exploration_rate'
        }
        
        for param, value in optional_params.items():
            if value is not None and param in param_flags:
                cmd.extend([param_flags[param], str(value)])
        
        # Handle boolean flags
        if config.allow_scaling:
            cmd.append('--allow_scaling')
        if config.dry_run:
            cmd.append('--dry_run')
        
        # Start inference process
        inference_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        inference_active = True
        
        # Start background task to monitor inference output
        background_tasks.add_task(monitor_inference_output)
        
        return {"message": "Inference started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop_inference")
async def stop_inference():
    global inference_process, inference_active
    
    if not inference_active or not inference_process:
        raise HTTPException(status_code=400, detail="No inference process is running")
    
    try:
        # Terminate the process
        inference_process.terminate()
        try:
            inference_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            inference_process.kill()
        
        inference_active = False
        
        await manager.broadcast({
            "type": "inference_output", 
            "data": "Inference stopped by user\n",
            "output_type": "info",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"message": "Inference stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list_files", response_model=FileList)
async def list_files(file_type: Optional[str] = "all"):
    """List available model and data files"""
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        files = {
            'models': [],
            'data': [],
            'scalers': []
        }
        
        if file_type in ['all', 'models']:
            # Look for model files
            model_paths = [
                os.path.join(base_path, 'models'),
                os.path.join(base_path, 'training')
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    for root, dirs, filenames in os.walk(model_path):
                        for filename in filenames:
                            if filename.endswith(('.pt', '.pth')):
                                full_path = os.path.join(root, filename)
                                rel_path = os.path.relpath(full_path, base_path)
                                files['models'].append(rel_path)
        
        if file_type in ['all', 'data']:
            # Look for data files
            data_paths = [
                os.path.join(base_path, 'data'),
                base_path
            ]
            
            for data_path in data_paths:
                if os.path.exists(data_path):
                    for root, dirs, filenames in os.walk(data_path):
                        for filename in filenames:
                            if filename.endswith(('.csv', '.parquet')):
                                full_path = os.path.join(root, filename)
                                rel_path = os.path.relpath(full_path, base_path)
                                files['data'].append(rel_path)
        
        if file_type in ['all', 'scalers']:
            # Look for scaler files
            scaler_paths = [
                os.path.join(base_path, 'models'),
                os.path.join(base_path, 'training'),
                base_path
            ]
            
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    for root, dirs, filenames in os.walk(scaler_path):
                        for filename in filenames:
                            if filename.endswith(('.joblib', '.pkl')):
                                full_path = os.path.join(root, filename)
                                rel_path = os.path.relpath(full_path, base_path)
                                files['scalers'].append(rel_path)
        
        return FileList(**files)
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back any messages (can be extended for bidirectional communication)
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background tasks for monitoring processes
async def monitor_training_output():
    global training_process, training_active
    
    if not training_process:
        return
    
    try:
        while training_process.poll() is None:
            line = training_process.stdout.readline()
            if line:
                # Parse training output for specific metrics
                output_type = 'info'
                if 'error' in line.lower() or 'exception' in line.lower():
                    output_type = 'error'
                elif 'episode' in line.lower() or 'reward' in line.lower():
                    output_type = 'metric'
                elif 'loss' in line.lower():
                    output_type = 'loss'
                
                await manager.broadcast({
                    "type": "training_output",
                    "data": line,
                    "output_type": output_type,
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error monitoring training output: {e}")
    finally:
        training_active = False
        await manager.broadcast({
            "type": "training_output",
            "data": "Training process finished\n",
            "output_type": "info",
            "timestamp": datetime.now().isoformat()
        })

async def monitor_inference_output():
    global inference_process, inference_active
    
    if not inference_process:
        return
    
    try:
        while inference_process.poll() is None:
            line = inference_process.stdout.readline()
            if line:
                # Parse inference output for specific metrics
                output_type = 'info'
                if 'error' in line.lower() or 'exception' in line.lower():
                    output_type = 'error'
                elif any(signal in line.lower() for signal in ['hold','buy', 'sell']):
                    output_type = 'signal'
                elif 'position' in line.lower():
                    output_type = 'position'
                elif 'price' in line.lower():
                    output_type = 'price'
                
                await manager.broadcast({
                    "type": "inference_output",
                    "data": line,
                    "output_type": output_type,
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error monitoring inference output: {e}")
    finally:
        inference_active = False
        await manager.broadcast({
            "type": "inference_output",
            "data": "Inference process finished\n",
            "output_type": "info",
            "timestamp": datetime.now().isoformat()
        })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )