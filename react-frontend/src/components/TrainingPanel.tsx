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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Checkbox,
  Alert,
  Snackbar,
  CircularProgress,

  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  PlayArrow,
  Stop,
  FolderOpen,
  Settings,
  DataUsage,
  Architecture,
  TuneOutlined,
} from '@mui/icons-material';
import axios from 'axios';
import LogOutput from './LogOutput';

interface TrainingConfig {
  train_data: string;
  symbol: string;
  interval: string;
  window_size: number;
  leverage: number;
  max_position: number;
  balance: number;
  risk_reward: number;
  stop_loss: number;
  trade_fee: number;
  architecture: string;
  embedding_dim: number;
  n_encoder_layers: number;
  n_decoder_layers: number;
  dropout: number;
  attention_type: string;
  n_heads: number;
  n_latents: number;
  n_groups: number;
  ffn_type: string;
  ffn_dim: number;
  n_experts: number;
  top_k: number;
  norm_type: string;
  feature_extractor_type: string;
  feature_extractor_dim: number;
  feature_extractor_layers: number;
  head_hidden_dim: number;
  head_n_layers: number;
  lr: number;
  gamma: number;
  gae_lambda: number;
  policy_clip: number;
  batch_size: number;
  n_epochs: number;
  entropy_coef: number;
  value_coef: number;
  max_grad_norm: number;
  weight_decay: number;
  episodes: number;
  n_splits: number;
  val_ratio: number;
  eval_freq: number;
  save_path: string;
  device: string;
  dynamic_leverage: boolean;
  use_risk_adjusted_rewards: boolean;
  use_skip_connections: boolean;
  use_layer_norm: boolean;
  use_instance_norm: boolean;
  head_use_layer_norm: boolean;
  head_use_residual: boolean;
  use_gae: boolean;
  normalize_advantage: boolean;
}

// Remove unused interface - FileList is available globally from DOM types

const TrainingPanel: React.FC = () => {
  const [config, setConfig] = useState<TrainingConfig>({
    train_data: '',
    symbol: 'BTCUSDT',
    interval: '15m',
    window_size: 24,
    leverage: 2,
    max_position: 1000,
    balance: 10000,
    risk_reward: 2.0,
    stop_loss: 0.02,
    trade_fee: 0.001,
    architecture: 'encoder_only',
    embedding_dim: 128,
    n_encoder_layers: 4,
    n_decoder_layers: 4,
    dropout: 0.1,
    attention_type: 'mha',
    n_heads: 8,
    n_latents: 32,
    n_groups: 4,
    ffn_type: 'standard',
    ffn_dim: 512,
    n_experts: 8,
    top_k: 2,
    norm_type: 'layer_norm',
    feature_extractor_type: 'basic',
    feature_extractor_dim: 256,
    feature_extractor_layers: 3,
    head_hidden_dim: 256,
    head_n_layers: 2,
    lr: 0.001,
    gamma: 0.99,
    gae_lambda: 0.95,
    policy_clip: 0.2,
    batch_size: 64,
    n_epochs: 10,
    entropy_coef: 0.01,
    value_coef: 0.5,
    max_grad_norm: 0.5,
    weight_decay: 0.0001,
    episodes: 1000,
    n_splits: 5,
    val_ratio: 0.2,
    eval_freq: 100,
    save_path: 'models/best_model.pt',
    device: 'auto',
    dynamic_leverage: true,
    use_risk_adjusted_rewards: true,
    use_skip_connections: false,
    use_layer_norm: false,
    use_instance_norm: false,
    head_use_layer_norm: false,
    head_use_residual: false,
    use_gae: true,
    normalize_advantage: true,
  });

  const [isTraining, setIsTraining] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as 'success' | 'error' | 'info' });

  useEffect(() => {
    const wsUrl = window.location.protocol === 'https:' 
      ? `wss://${window.location.host}/ws`
      : `ws://${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'training_output') {
        setLogs(prev => [...prev, `[${data.timestamp}] ${data.data}`]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event);
    };

    return () => {
      ws.close();
    };
  }, []);

  const fetchFiles = async () => {
    // Placeholder function for file browsing
    console.log('File browsing not implemented yet');
  };

  const handleInputChange = (field: keyof TrainingConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const startTraining = async () => {
    try {
      setIsTraining(true);
      setLogs([]);
      await axios.post('/api/start_training', config);
      setSnackbar({ open: true, message: 'Training started successfully!', severity: 'success' });
    } catch (error: any) {
      setIsTraining(false);
      setSnackbar({ 
        open: true, 
        message: error.response?.data?.error || 'Failed to start training', 
        severity: 'error' 
      });
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('/api/stop_training');
      setIsTraining(false);
      setSnackbar({ open: true, message: 'Training stopped successfully!', severity: 'info' });
    } catch (error: any) {
      setSnackbar({ 
        open: true, 
        message: error.response?.data?.error || 'Failed to stop training', 
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
                <Typography variant="h6">Training Configuration</Typography>
              </Box>

              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <DataUsage sx={{ mr: 1 }} />
                    <Typography>Data & Environment</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid size={12}>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <TextField
                          fullWidth
                          label="Training Data Path *"
                          value={config.train_data}
                          onChange={(e) => handleInputChange('train_data', e.target.value)}
                          required
                        />
                        <Tooltip title="Browse Files">
                          <IconButton onClick={fetchFiles}>
                            <FolderOpen />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Grid>
                    <Grid size={6}>
                      <FormControl fullWidth>
                        <InputLabel>Symbol *</InputLabel>
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
                    <Grid size={4}>
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
                    <Grid size={4}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Window Size"
                        value={config.window_size}
                        onChange={(e) => handleInputChange('window_size', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 1000 }}
                      />
                    </Grid>
                    <Grid size={4}>
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
                        label="Initial Balance"
                        value={config.balance}
                        onChange={(e) => handleInputChange('balance', parseFloat(e.target.value))}
                        inputProps={{ min: 100, step: 0.01 }}
                      />
                    </Grid>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Risk Reward Ratio"
                        value={config.risk_reward}
                        onChange={(e) => handleInputChange('risk_reward', parseFloat(e.target.value))}
                        inputProps={{ min: 0.1, max: 10, step: 0.1 }}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Architecture sx={{ mr: 1 }} />
                    <Typography>Model Architecture</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                  <Grid size={12}>
                    <FormControl fullWidth>
                      <InputLabel>Feature Extractor</InputLabel>
                      <Select
                        value={config.feature_extractor_type}
                        onChange={(e) => handleInputChange('feature_extractor_type', e.target.value)}
                      >
                        <MenuItem value="basic">Basic</MenuItem>
                        <MenuItem value="resnet">ResNet</MenuItem>
                        <MenuItem value="inception">Inception</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                    <Grid size={6}>
                      <FormControl fullWidth>
                        <InputLabel>Architecture</InputLabel>
                        <Select
                          value={config.architecture}
                          onChange={(e) => handleInputChange('architecture', e.target.value)}
                        >
                          <MenuItem value="encoder_only">Encoder Only</MenuItem>
                          <MenuItem value="decoder_only">Decoder Only</MenuItem>
                          <MenuItem value="encoder_decoder">Encoder-Decoder</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Embedding Dimension"
                        value={config.embedding_dim}
                        onChange={(e) => handleInputChange('embedding_dim', parseInt(e.target.value))}
                        inputProps={{ min: 32, max: 1024 }}
                      />
                    </Grid>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Encoder Layers"
                        value={config.n_encoder_layers}
                        onChange={(e) => handleInputChange('n_encoder_layers', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 24 }}
                      />
                    </Grid>
                    <Grid size={6}>
                      <FormControl fullWidth>
                        <InputLabel>Attention Type</InputLabel>
                        <Select
                          value={config.attention_type}
                          onChange={(e) => handleInputChange('attention_type', e.target.value)}
                        >
                          <MenuItem value="mha">Multi-Head Attention</MenuItem>
                          <MenuItem value="pyramidal">Pyramidal Attention</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid size={6}>
                      <FormControl fullWidth>
                        <InputLabel>Feed Forward Type</InputLabel>
                        <Select
                          value={config.ffn_type}
                          onChange={(e) => handleInputChange('ffn_type', e.target.value)}
                        >
                          <MenuItem value="standard">Standard</MenuItem>
                          <MenuItem value="moe">Mixture of Experts</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>

                    
                    {config.ffn_type === 'moe' && (
                      <>
                        <Grid size={6}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Number of Experts"
                            value={config.n_experts}
                            onChange={(e) => handleInputChange('n_experts', parseInt(e.target.value))}
                            inputProps={{ min: 2, max: 32 }}
                          />
                        </Grid>
                        <Grid size={6}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Top K Experts"
                            value={config.top_k}
                            onChange={(e) => handleInputChange('top_k', parseInt(e.target.value))}
                            inputProps={{ min: 1, max: 8 }}
                          />
                        </Grid>
                      </>
                    )}

                    <Grid size={12}>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={config.use_skip_connections}
                              onChange={(e) => handleInputChange('use_skip_connections', e.target.checked)}
                            />
                          }
                          label="Skip Connections"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={config.use_layer_norm}
                              onChange={(e) => handleInputChange('use_layer_norm', e.target.checked)}
                            />
                          }
                          label="Layer Norm"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={config.dynamic_leverage}
                              onChange={(e) => handleInputChange('dynamic_leverage', e.target.checked)}
                            />
                          }
                          label="Dynamic Leverage"
                        />
                      </Box>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <TuneOutlined sx={{ mr: 1 }} />
                    <Typography>Training Parameters</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Learning Rate"
                        value={config.lr}
                        onChange={(e) => handleInputChange('lr', parseFloat(e.target.value))}
                        inputProps={{ min: 0.0001, max: 0.1, step: 0.0001 }}
                      />
                    </Grid>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Batch Size"
                        value={config.batch_size}
                        onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 512 }}
                      />
                    </Grid>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Episodes"
                        value={config.episodes}
                        onChange={(e) => handleInputChange('episodes', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 10000 }}
                      />
                    </Grid>
                    <Grid size={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="N Epochs"
                        value={config.n_epochs}
                        onChange={(e) => handleInputChange('n_epochs', parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 100 }}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={isTraining ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
                  onClick={startTraining}
                  disabled={isTraining || !config.train_data || !config.symbol}
                  sx={{ flex: 1 }}
                >
                  {isTraining ? 'Training...' : 'Start Training'}
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  onClick={stopTraining}
                  disabled={!isTraining}
                  sx={{ flex: 1 }}
                >
                  Stop Training
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, lg: 6 }}>
          <LogOutput title="Training Output" logs={logs} />
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

export default TrainingPanel;