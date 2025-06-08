import React, { useState } from 'react';
import axios from 'axios';
import {
  Box,
  Grid,
  TextField,
  Button,
  Checkbox,
  FormControlLabel,
  Snackbar,
  Alert,
  Typography,
  Link,
} from '@mui/material';
import LogOutput from './LogOutput';

interface DataConfig {
  symbol: string;
  interval: string;
  days: number;
  start_date?: string;
  end_date?: string;
  output_dir: string;
  split: boolean;
  test_ratio: number;
  validation_ratio: number;
}

const DataPanel: React.FC = () => {
  const [config, setConfig] = useState<DataConfig>({
    symbol: 'BTCUSDT',
    interval: '1h',
    days: 1,
    start_date: '',
    end_date: '',
    output_dir: 'data',
    split: false,
    test_ratio: 0.2,
    validation_ratio: 0.0,
  });
  const [logs, setLogs] = useState<string[]>([]);
  const [filePath, setFilePath] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error';
  }>({ open: false, message: '', severity: 'success' });

  const handleChangeField = (
    key: keyof DataConfig,
    value: string | number | boolean
  ) => {
    setConfig((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const processData = async () => {
    setLoading(true);
    setLogs([]);
    try {
      const response = await axios.post('/api/process_data', config);
      const output = response.data.output as string;
      setLogs(output.split('\n'));
      setSnackbar({
        open: true,
        message: response.data.message || 'Data processed successfully',
        severity: 'success',
      });
      setFilePath(response.data.file_path || '');
    } catch (error: any) {
      const msg =
        error.response?.data?.detail || error.message || 'Error processing data';
      setSnackbar({ open: true, message: msg, severity: 'error' });
      setFilePath('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={4}>
          <TextField
            label="Symbol"
            fullWidth
            value={config.symbol}
            onChange={(e) => handleChangeField('symbol', e.target.value.toUpperCase())}
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <TextField
            label="Interval"
            fullWidth
            value={config.interval}
            onChange={(e) => handleChangeField('interval', e.target.value)}
            helperText="e.g. 1m, 5m, 15m, 1h, 1d"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <TextField
            label="Days"
            type="number"
            fullWidth
            value={config.days}
            onChange={(e) => handleChangeField('days', Number(e.target.value))}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            label="Start Date"
            type="date"
            fullWidth
            InputLabelProps={{ shrink: true }}
            value={config.start_date}
            onChange={(e) => handleChangeField('start_date', e.target.value)}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            label="End Date"
            type="date"
            fullWidth
            InputLabelProps={{ shrink: true }}
            value={config.end_date}
            onChange={(e) => handleChangeField('end_date', e.target.value)}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            label="Output Directory"
            fullWidth
            value={config.output_dir}
            onChange={(e) => handleChangeField('output_dir', e.target.value)}
            helperText="Directory to save processed data"
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControlLabel
            control={
              <Checkbox
                checked={config.split}
                onChange={(e) => handleChangeField('split', e.target.checked)}
              />
            }
            label="Split into train/test"
          />
        </Grid>
        {config.split && (
          <>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Test Ratio"
                type="number"
                fullWidth
                inputProps={{ step: 0.01 }}
                value={config.test_ratio}
                onChange={(e) =>
                  handleChangeField('test_ratio', Number(e.target.value))
                }
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Validation Ratio"
                type="number"
                fullWidth
                inputProps={{ step: 0.01 }}
                value={config.validation_ratio}
                onChange={(e) =>
                  handleChangeField('validation_ratio', Number(e.target.value))
                }
              />
            </Grid>
          </>
        )}
      </Grid>

      <Box mt={2}>
        <Button
          variant="contained"
          color="primary"
          onClick={processData}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Process Data'}
        </Button>
      </Box>

      {filePath && (
        <Box mb={2}>
          <Typography variant="body2">
            File saved at: <Link href={filePath} target="_blank" rel="noopener">{filePath}</Link>
          </Typography>
        </Box>
      )}
      <Box mt={3}>
        <LogOutput title="Data Output" logs={logs} />
      </Box>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataPanel;