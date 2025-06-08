import React, { useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Clear,
  Download,
} from '@mui/icons-material';

interface LogOutputProps {
  title: string;
  logs: string[];
}

const LogOutput: React.FC<LogOutputProps> = ({ title, logs }) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const clearLogs = () => {
    // This would need to be passed down as a prop or handled by parent
    console.log('Clear logs');
  };

  const downloadLogs = () => {
    const logText = logs.join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(' ', '_')}_logs.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getLogColor = (log: string) => {
    const lowercaseLog = log.toLowerCase();
    if (lowercaseLog.includes('error') || lowercaseLog.includes('exception')) {
      return '#f44336';
    } else if (lowercaseLog.includes('warning') || lowercaseLog.includes('warn')) {
      return '#ff9800';
    } else if (lowercaseLog.includes('episode') || lowercaseLog.includes('reward')) {
      return '#4caf50';
    } else if (lowercaseLog.includes('buy') || lowercaseLog.includes('sell')) {
      return '#2196f3';
    }
    return 'inherit';
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ pb: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">{title}</Typography>
          <Box>
            <Tooltip title="Download Logs">
              <IconButton onClick={downloadLogs} size="small">
                <Download />
              </IconButton>
            </Tooltip>
            <Tooltip title="Clear Logs">
              <IconButton onClick={clearLogs} size="small">
                <Clear />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </CardContent>
      
      <Box sx={{ flexGrow: 1, px: 2, pb: 2 }}>
        <Paper
          ref={logContainerRef}
          sx={{
            height: '500px',
            overflow: 'auto',
            p: 2,
            backgroundColor: '#000',
            fontFamily: 'Monaco, Consolas, "Courier New", monospace',
            fontSize: '0.875rem',
            lineHeight: 1.4,
          }}
        >
          {logs.length === 0 ? (
            <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>
              No logs yet...
            </Typography>
          ) : (
            logs.map((log, index) => (
              <Box
                key={index}
                sx={{
                  color: getLogColor(log),
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  mb: 0.5,
                }}
              >
                {log}
              </Box>
            ))
          )}
        </Paper>
      </Box>
    </Card>
  );
};

export default LogOutput;