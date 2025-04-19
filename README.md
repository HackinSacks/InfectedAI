# InfectedAI

InfectedAI is an advanced, AI-driven penetration testing tool that automates the process of vulnerability scanning and exploitation using the Metasploit Framework. Leveraging reinforcement learning and a simulated language model, InfectedAI intelligently selects and executes exploits, making penetration testing more efficient for security professionals. This tool is designed for ethical use in controlled environments with explicit permission.

## Features
- **Automated Scanning**: Uses Nmap to identify target system characteristics (OS, open ports, services).
- **AI-Driven Exploit Selection**: Combines reinforcement learning (Deep Q-Learning) and a simulated language model to choose optimal exploits.
- **Metasploit Integration**: Interacts with Metasploit via its RPC API to execute exploits and manage sessions.
- **Extensible Framework**: Placeholder for integrating a fine-tuned large language model (LLM) to enhance decision-making and script generation.
- **Ethical Design**: Built with responsible use in mind, requiring explicit authorization for testing.

## Prerequisites
Before using InfectedAI, ensure you have the following setup:
- **Operating System**: Kali Linux (recommended for penetration testing tools).
- **Metasploit Framework**: Installed and configured ([Metasploit Download](https://www.metasploit.com/download)).
- **Python 3.8+**: Required for running the tool and its dependencies.
- **Nmap**: For target scanning (`sudo apt-get install nmap`).
- **Test Environment**: A virtualized network with vulnerable machines (e.g., Metasploitable 2/3) for safe testing.

## Installation
Follow these steps to set up InfectedAI on your system:

### 1. Clone the Repository
```bash
git clone https://github.com/GulfOfAmerica/InfectedAI.git
cd InfectedAI
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install python-nmap tensorflow msfrpc
```

### 3. Configure Metasploit RPC
Start the Metasploit RPC server to allow InfectedAI to interact with it:
```bash
msfrpcd -P your_password -U your_username -S
```
Replace `your_password` and `your_username` with your desired credentials.

### 4. Update Configuration
Edit the `ai_pentest_tool.py` script to set your target IP and Metasploit RPC credentials:
- Update `target_ip` to the IP address of your test target (e.g., a Metasploitable VM).
- Replace `your_password` and `your_username` with the credentials used in step 3.

## Usage
Run InfectedAI to perform an automated penetration test on your target system:
```bash
python ai_pentest_tool.py
```

### How It Works
1. **Target Scanning**: InfectedAI uses Nmap to gather information about the target (OS, ports, services).
2. **Exploit Selection**:
   - A simulated language model suggests exploits based on predefined rules.
   - A reinforcement learning agent (Deep Q-Learning) selects exploits if the language model fails to provide a suggestion.
3. **Exploit Execution**: The tool executes the selected exploit via Metasploit and checks for successful exploitation.
4. **Learning**: The RL agent learns from each attempt, improving its exploit selection over time.

### Example Output
```
2025-04-19 10:00:00,123 - INFO - Starting episode 1
2025-04-19 10:00:01,456 - INFO - Scanned target 192.168.1.100: {'os': 'Linux', 'services': [{'port': 21, 'info': 'vsftpd 2.3.4'}]}
2025-04-19 10:00:02,789 - INFO - LLM suggested exploit: exploit/unix/ftp/vsftpd_234_backdoor
2025-04-19 10:00:08,234 - INFO - Exploit exploit/unix/ftp/vsftpd_234_backdoor succeeded on 192.168.1.100
2025-04-19 10:00:08,235 - INFO - Successful exploitation, stopping episode
```

## Ethical and Legal Considerations
InfectedAI is a powerful tool that must be used responsibly:
- **Permission**: Only test systems you own or have explicit written authorization to test. Unauthorized testing is illegal under laws like the Computer Fraud and Abuse Act (CFAA).
- **Safe Environment**: Use a virtualized test network (e.g., VirtualBox with Metasploitable) to avoid harming real systems.
- **Data Privacy**: Do not use sensitive or unauthorized data for training or testing.
- **Responsible Disclosure**: Document and report findings transparently to stakeholders.

## Extending with a Large Language Model (LLM)
InfectedAI includes a placeholder for integrating a fine-tuned LLM to enhance exploit selection and script generation. Follow these steps to extend the tool:

### 1. Select an LLM
Choose a model like LLaMA (research license required) or an open-source alternative (e.g., Mistral).

### 2. Collect Training Data
Gather penetration testing data ethically:
- Simulated data from test environments.
- Public vulnerability databases (e.g., CVE, NVD).
- Format as prompt-response pairs (e.g., "Target: Linux, Port 21, vsftpd 2.3.4. Suggest an exploit." → "Use exploit/unix/ftp/vsftpd_234_backdoor.").

### 3. Fine-Tune the LLM
Use a framework like Hugging Face Transformers:
- Preprocess data into JSONL format.
- Fine-tune with 4–8 GPUs (e.g., NVIDIA A100) for 1–2 days.
- Example command:
  ```bash
  python -m transformers.run_clm --model_name_or_path meta-llama/LLaMA-7B --train_file dataset.jsonl --output_dir fine_tuned_model --num_train_epochs 3
  ```

### 4. Integrate the LLM
Replace the `SimulatedLLM` class in `ai_pentest_tool.py` with your fine-tuned model:
```python
from transformers import pipeline
llm = pipeline('text-generation', model='path/to/fine_tuned_model')
def interpret_scan(self, target_info):
    prompt = f"Target: {target_info['os']}, Ports: {target_info['services']}. Suggest an exploit."
    response = llm(prompt, max_length=100)
    return response[0]['generated_text']
```

## Limitations
- **Simulated LLM**: The current implementation uses a rule-based system instead of a fine-tuned LLM, limiting its intelligence.
- **Resource Intensive**: Fine-tuning an LLM requires significant computational resources.
- **Simplified RL**: The reinforcement learning model uses a basic state representation and may not handle complex scenarios.
- **Maintenance**: Must be updated with new exploits as Metasploit evolves.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to your branch (`git push -u origin feature-branch`).
5. Open a pull request.

## License
InfectedAI is dual-licensed:
- **For Non-Commercial Use**: Licensed under the InfectedAI Open License, which allows free use, modification, and distribution for non-commercial purposes. See the [LICENSE](LICENSE) file for details.
- **For Commercial Use**: A separate commercial license is required. Contact [GulfOfAmerica](https://github.com/GulfOfAmerica) for pricing and terms.

## Acknowledgments
- [Metasploit Framework](https://www.metasploit.com/) for the penetration testing foundation.
- [Nmap](https://nmap.org/) for target scanning capabilities.
- [TensorFlow](https://www.tensorflow.org/) for reinforcement learning implementation.

## Contact
For questions, support, or commercial licensing inquiries, open an issue on GitHub or contact the maintainers at [GulfOfAmerica](https://github.com/GulfOfAmerica).

---
**Warning**: InfectedAI is for educational and ethical use only. Misuse of this tool for unauthorized or malicious purposes is strictly prohibited and may result in legal consequences.
