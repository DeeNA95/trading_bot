import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import {
  School,
  Timeline,
  DataUsage,
} from '@mui/icons-material';
import TrainingPanel from './TrainingPanel';
import InferencePanel from './InferencePanel';
import DataPanel from './DataPanel';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

const Dashboard: React.FC = () => {
  const [value, setValue] = useState(0);

  const handleChange = (_event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={1} sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="trading bot tabs"
            centered
            sx={{
              '& .MuiTab-root': {
                minWidth: 200,
                fontSize: '1.1rem',
                fontWeight: 500,
                textTransform: 'none',
              },
            }}
          >
            <Tab
              icon={<DataUsage />}
              iconPosition="start"
              label="Data"
              {...a11yProps(0)}
            />
            <Tab
              icon={<School />}
              iconPosition="start"
              label="Training"
              {...a11yProps(1)}
            />
            <Tab
              icon={<Timeline />}
              iconPosition="start"
              label="Inference"
              {...a11yProps(2)}
            />
          </Tabs>
        </Box>
      </Paper>

      <TabPanel value={value} index={0}>
        <DataPanel />
      </TabPanel>
      <TabPanel value={value} index={1}>
        <TrainingPanel />
      </TabPanel>
      <TabPanel value={value} index={2}>
        <InferencePanel />
      </TabPanel>
    </Box>
  );
};

export default Dashboard;