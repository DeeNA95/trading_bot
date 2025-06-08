import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  SmartToy,
  Refresh,
} from '@mui/icons-material';
import axios from 'axios';

interface StatusData {
  training_active: boolean;
  inference_active: boolean;
}

const Navbar: React.FC = () => {
  const [status, setStatus] = useState<StatusData>({
    training_active: false,
    inference_active: false,
  });

  const fetchStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      setStatus(response.data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <AppBar position="static" sx={{ bgcolor: 'background.paper', color: 'text.primary' }}>
      <Toolbar>
        <SmartToy sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Trading Bot Control Panel
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            label={`Training: ${status.training_active ? 'Running' : 'Stopped'}`}
            color={status.training_active ? 'success' : 'default'}
            variant={status.training_active ? 'filled' : 'outlined'}
            size="small"
          />
          <Chip
            label={`Inference: ${status.inference_active ? 'Running' : 'Stopped'}`}
            color={status.inference_active ? 'success' : 'default'}
            variant={status.inference_active ? 'filled' : 'outlined'}
            size="small"
          />
          <Tooltip title="Refresh Status">
            <IconButton onClick={fetchStatus} size="small">
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;