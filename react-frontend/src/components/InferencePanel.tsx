import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  FormControlLabel,
  Checkbox,
  Alert,
  Snackbar,
  CircularProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  FolderOpen,
  Settings,
} from '@mui/icons-material';
import axios from 'axios';
import LogOutput from './LogOutput';

interface InferenceConfig {
  model_path: string;
  scaler_path: string;
  symbol: string;
  interval: string;
  window_size: number;
  leverage: number;
  risk_reward_ratio: number;
  stop_loss_percent: number;
  initial_balance: number;
  base_url: string;
  sleep_time: number;
  device: string;
  exploration_rate: number;
  allow_scaling: boolean;
  dry_run: boolean;
}

interface FileList {
  models: string[];
  data: string[];
  scalers: string[];
}

const InferencePanel: React.FC = () => {
  const [config, setConfig] = useState<InferenceConfig>({
    model_path: '',
    scaler_path: '',
    symbol: 'BTCUSDT',
    interval: '15m',
    window_size: 24,
    leverage: 2,
    risk_reward_ratio: 2.0,
    stop_loss_percent: 0.02,
    initial_balance: 10000,
    base_url: 'https://api.binance.com',
    sleep_time: 60,
    device: 'auto',
    exploration_rate: 0.1,
    allow_scaling: false,
    dry_run: true,
  });

  const [files, setFiles] = useState<FileList>({ models: [], data: [], scalers: [] });
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as 'success' | 'error' | 'info' });
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3000/ws');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'inference_output') {
        setLogs(prev => [...prev, `[${data.timestamp}] ${data.data}`]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    fetchFiles();

    return () => {
      ws.close();
    };
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await axios.get('/api/list_files');
      setFiles(response.data);
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  const handleInputChange = (field: keyof InferenceConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const startInference = async () => {
    try {
      setIsRunning(true);
      setLogs([]);
      await axios.post('/api/start_inference', config);
      setSnackbar({ open: true, message: 'Inference started successfully!', severity: 'success' });
    } catch (error: any) {
      setIsRunning(false);
      setSnackbar({ 
        open: true, 
        message: error.response?.data?.error || 'Failed to start inference', 
        severity: 'error' 
      });
    }
  };

  const stopInference = async () => {
    try {
      await axios.post('/api/stop_inference');
      setIsRunning(false);
      setSnackbar({ open: true, message: 'Inference stopped successfully!', severity: 'info' });
    } catch (error: any) {
      setSnackbar({ 
        open: true, 
        message: error.response?.data?.error || 'Failed to stop inference', 
        severity: 'error' 
      });
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, lg: 6 }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Settings sx={{ mr: 1 }} />
                <Typography variant="h6">Inference Configuration</Typography>
              </Box>

              <Grid container spacing={2}>
                <Grid size={12}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <FormControl fullWidth>
                      <InputLabel>Model Path *</InputLabel>
                      <Select
                        value={config.model_path}
                        onChange={(e) => handleInputChange('model_path', e.target.value)}
                        required
                      >
                        {files.models.map((model) => (
                          <MenuItem key={model} value={model}>
                            {model}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Tooltip title="Browse Files">
                      <IconButton onClick={fetchFiles}>
                        <FolderOpen />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Grid>

                <Grid size={12}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <FormControl fullWidth>
                      <InputLabel>Scaler Path *</InputLabel>
                      <Select
                        value={config.scaler_path}
                        onChange={(e) => handleInputChange('scaler_path', e.target.value)}
                        required
                      >
                        {files.scalers.map((scaler) => (
                          <MenuItem key={scaler} value={scaler}>
                            {scaler}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Tooltip title="Browse Files">
                      <IconButton onClick={fetchFiles}>
                        <FolderOpen />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Grid>

                <Grid size={6}>
                  <FormControl fullWidth>
                    <InputLabel>Symbol</InputLabel>
                    <Select
                      value={config.symbol}
                      onChange={(e) => handleInputChange('symbol', e.target.value)}
                    >
                      <MenuItem value="BTCUSDT">BTCUSDT</MenuItem>
                      <MenuItem value="ETHUSDT">ETHUSDT</MenuItem>
                      <MenuItem value="ADAUSDT">ADAUSDT</MenuItem>
                      <MenuItem value="BNBUSDT">BNBUSDT</MenuItem>
                      <MenuItem value="SOLUSDT">SOLUSDT</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid size={6}>
                  <FormControl fullWidth>
                    <InputLabel>Interval</InputLabel>
                    <Select
                      value={config.interval}
                      onChange={(e) => handleInputChange('interval', e.target.value)}
                    >
                      <MenuItem value="1m">1m</MenuItem>
                      <MenuItem value="3m">3m</MenuItem>
                      <MenuItem value="5m">5m</MenuItem>
                      <MenuItem value="15m">15m</MenuItem>
                      <MenuItem value="30m">30m</MenuItem>
                      <MenuItem value="1h">1h</MenuItem>
                      <MenuItem value="4h">4h</MenuItem>
                      <MenuItem value="1d">1d</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Window Size"
                    value={config.window_size}
                    onChange={(e) => handleInputChange('window_size', parseInt(e.target.value))}
                    inputProps={{ min: 1, max: 1000 }}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Leverage"
                    value={config.leverage}
                    onChange={(e) => handleInputChange('leverage', parseInt(e.target.value))}
                    inputProps={{ min: 1, max: 125 }}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Risk Reward Ratio"
                    value={config.risk_reward_ratio}
                    onChange={(e) => handleInputChange('risk_reward_ratio', parseFloat(e.target.value))}
                    inputProps={{ min: 0.1, max: 10, step: 0.1 }}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Stop Loss %"
                    value={config.stop_loss_percent}
                    onChange={(e) => handleInputChange('stop_loss_percent', parseFloat(e.target.value))}
                    inputProps={{ min: 0.001, max: 1, step: 0.001 }}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Initial Balance"
                    value={config.initial_balance}
                    onChange={(e) => handleInputChange('initial_balance', parseFloat(e.target.value))}
                    inputProps={{ min: 100, step: 0.01 }}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Sleep Time (seconds)"
                    value={config.sleep_time}
                    onChange={(e) => handleInputChange('sleep_time', parseInt(e.target.value))}
                    inputProps={{ min: 1, max: 3600 }}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Exploration Rate"
                    value={config.exploration_rate}
                    onChange={(e) => handleInputChange('exploration_rate', parseFloat(e.target.value))}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                  />
                </Grid>

                <Grid size={6}>
                  <FormControl fullWidth>
                    <InputLabel>Device</InputLabel>
                    <Select
                      value={config.device}
                      onChange={(e) => handleInputChange('device', e.target.value)}
                    >
                      <MenuItem value="auto">Auto</MenuItem>
                      <MenuItem value="cpu">CPU</MenuItem>
                      <MenuItem value="cuda">CUDA</MenuItem>
                      <MenuItem value="mps">MPS</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid size={12}>
                  <TextField
                    fullWidth
                    label="Base URL"
                    value={config.base_url}
                    onChange={(e) => handleInputChange('base_url', e.target.value)}
                  />
                </Grid>

                <Grid size={12}>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={config.dry_run}
                          onChange={(e) => handleInputChange('dry_run', e.target.checked)}
                        />
                      }
                      label="Dry Run (No Real Trades)"
                    />
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={config.allow_scaling}
                          onChange={(e) => handleInputChange('allow_scaling', e.target.checked)}
                        />
                      }
                      label="Allow Scaling"
                    />
                  </Box>
                </Grid>
              </Grid>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
                  onClick={startInference}
                  disabled={isRunning || !config.model_path || !config.scaler_path}
                  sx={{ flex: 1 }}
                >
                  {isRunning ? 'Running...' : 'Start Inference'}
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  onClick={stopInference}
                  disabled={!isRunning}
                  sx={{ flex: 1 }}
                >
                  Stop Inference
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, lg: 6 }}>
          <LogOutput title="Inference Output" logs={logs} />
        </Grid>
      </Grid>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default InferencePanel;