#!/usr/bin/env python3
"""
Genyx AI - Sistema de IA Autônomo para Termux
Versão: 2.0.0 - Qwen 2.5 via Requests Puro
Autor: Genyx Development Team
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, Dict
import requests

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

class Config:
    """Configuração central do sistema"""
    
    API_ENDPOINT = "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-7B-Instruct"
    API_TIMEOUT = 120
    
    BRAIN_HISTORY = Path("brain_history.json")
    LEARNING_LOG = Path("learning_log.txt")
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_CYAN = "\033[96m"

# ============================================================================
# SISTEMA DE PERSISTÊNCIA
# ============================================================================

class BrainHistory:
    """Gerenciador de histórico e estatísticas"""
    
    def __init__(self, filepath: Path = Config.BRAIN_HISTORY):
        self.filepath = filepath
        self.data = self._load()
    
    def _load(self) -> Dict:
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._create_default()
        return self._create_default()
    
    def _create_default(self) -> Dict:
        return {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "version": "2.0.0",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "total_interactions": 0
            },
            "statistics": {
                "prompts_generated": 0,
                "mindmaps_created": 0,
                "sessions": 0
            },
            "interactions": []
        }
    
    def save(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def add_interaction(self, interaction_type: str, prompt: str, response: str):
        self.data["interactions"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": interaction_type,
            "prompt": prompt[:200],
            "response": response[:500]
        })
        
        self.data["metadata"]["total_interactions"] += 1
        
        if interaction_type == "prompt":
            self.data["statistics"]["prompts_generated"] += 1
        elif interaction_type == "mindmap":
            self.data["statistics"]["mindmaps_created"] += 1
        
        if len(self.data["interactions"]) > 100:
            self.data["interactions"] = self.data["interactions"][-100:]
        
        self.save()
    
    def get_stats(self) -> Dict:
        return {
            "total": self.data["metadata"]["total_interactions"],
            "prompts": self.data["statistics"]["prompts_generated"],
            "mindmaps": self.data["statistics"]["mindmaps_created"],
            "sessions": self.data["statistics"]["sessions"]
        }
    
    def increment_session(self):
        self.data["statistics"]["sessions"] += 1
        self.save()

# ============================================================================
# SISTEMA DE LOGGING
# ============================================================================

class LearningLogger:
    """Logger para treinamento futuro"""
    
    def __init__(self, filepath: Path = Config.LEARNING_LOG):
        self.filepath = filepath
    
    def log(self, interaction_type: str, user_input: str, ai_output: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
{'='*70}
[{timestamp}] TYPE: {interaction_type}
{'='*70}

USER INPUT:
{user_input}

AI OUTPUT:
{ai_output}

{'='*70}

"""
        
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except:
            pass
    
    def log_event(self, event: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] SYSTEM: {event}\n")
        except:
            pass

# ============================================================================
# CLIENTE API QWEN
# ============================================================================

class QwenClient:
    """Cliente para Qwen 2.5 via requests"""
    
    def __init__(self):
        self.endpoint = Config.API_ENDPOINT
        self.token = os.environ.get("HF_TOKEN")
        
        if not self.token:
            print(f"{Config.RED}❌ ERRO: HF_TOKEN não encontrado!{Config.RESET}")
            print(f"{Config.YELLOW}Configure: export HF_TOKEN=seu_token{Config.RESET}")
            sys.exit(1)
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
        """Gera texto usando Qwen 2.5"""
        
        print(f"{Config.CYAN}🔄 Enviando para Qwen 2.5...{Config.RESET}")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=Config.API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    text = result.get("generated_text", "")
                else:
                    text = str(result)
                
                if text:
                    print(f"{Config.GREEN}✅ Resposta recebida{Config.RESET}")
                    return text
                else:
                    print(f"{Config.RED}❌ Resposta vazia{Config.RESET}")
                    return None
            
            elif response.status_code == 503:
                print(f"{Config.YELLOW}⏳ Modelo carregando... Aguarde 20s{Config.RESET}")
                return None
            
            elif response.status_code == 401:
                print(f"{Config.RED}❌ Token inválido{Config.RESET}")
                return None
            
            elif response.status_code == 429:
                print(f"{Config.YELLOW}⚠️  Rate limit{Config.RESET}")
                return None
            
            else:
                print(f"{Config.RED}❌ HTTP {response.status_code}{Config.RESET}")
                return None
        
        except requests.Timeout:
            print(f"{Config.RED}❌ Timeout{Config.RESET}")
            return None
        
        except Exception as e:
            print(f"{Config.RED}❌ Erro: {e}{Config.RESET}")
            return None

# ============================================================================
# INTERFACE
# ============================================================================

class UI:
    """Interface de usuário"""
    
    @staticmethod
    def print_banner():
        banner = f"""
{Config.BRIGHT_CYAN}{Config.BOLD}
╔══════════════════════════════════════════════════╗
║  ██████╗ ███████╗███╗   ██╗██╗   ██╗██╗  ██╗   ║
║ ██╔════╝ ██╔════╝████╗  ██║╚██╗ ██╔╝╚██╗██╔╝   ║
║ ██║  ███╗█████╗  ██╔██╗ ██║ ╚████╔╝  ╚███╔╝    ║
║ ██║   ██║██╔══╝  ██║╚██╗██║  ╚██╔╝   ██╔██╗    ║
║ ╚██████╔╝███████╗██║ ╚████║   ██║   ██╔╝ ██╗   ║
║                                                  ║
║         AI System - Powered by Qwen 2.5         ║
║               Version 2.0.0 🚀                   ║
╚══════════════════════════════════════════════════╝
{Config.RESET}
"""
        print(banner)
    
    @staticmethod
    def print_menu(stats: Dict):
        print(f"\n{Config.BOLD}{'─'*50}{Config.RESET}")
        print(f"{Config.YELLOW}📊 Stats:{Config.RESET} "
              f"Total: {stats['total']} | "
              f"Prompts: {stats['prompts']} | "
              f"Mapas: {stats['mindmaps']}")
        print(f"{Config.BOLD}{'─'*50}{Config.RESET}\n")
        
        print(f"{Config.BRIGHT_GREEN}[1]{Config.RESET} 📝 Gerar Prompt Estruturado")
        print(f"{Config.BRIGHT_GREEN}[2]{Config.RESET} 🧠 Criar Mapa Mental")
        print(f"{Config.BRIGHT_GREEN}[3]{Config.RESET} 📈 Ver Estatísticas")
        print(f"{Config.BRIGHT_GREEN}[4]{Config.RESET} 📚 Sobre")
        print(f"{Config.BRIGHT_GREEN}[0]{Config.RESET} 🚪 Sair\n")

# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

class GenyxAI:
    """Sistema principal"""
    
    def __init__(self):
        self.brain = BrainHistory()
        self.logger = LearningLogger()
        self.client = QwenClient()
        self.ui = UI()
        
        self.brain.increment_session()
        self.logger.log_event("SYSTEM_START")
    
    def generate_prompt(self):
        """Gerar prompt estruturado"""
        print(f"\n{Config.CYAN}{'─'*50}")
        print(f"  📝 GERADOR DE PROMPT ESTRUTURADO")
        print(f"{'─'*50}{Config.RESET}\n")
        
        topic = input(f"{Config.YELLOW}🎯 Tema:{Config.RESET} ").strip()
        
        if not topic:
            print(f"{Config.RED}❌ Vazio{Config.RESET}")
            return
        
        template = f"""You are an expert prompt engineer. Create a detailed professional prompt for: {topic}

Include:
1. OBJECTIVE: Main goal
2. CONTEXT: Background
3. REQUIREMENTS: Specific elements
4. CONSTRAINTS: Limitations
5. OUTPUT FORMAT: Structure
6. STYLE: Tone
7. EXAMPLES: Scenarios

Make it actionable and specific."""
        
        result = self.client.generate(template, max_tokens=800, temperature=0.7)
        
        if result:
            print(f"\n{Config.BRIGHT_CYAN}╔{'═'*48}╗{Config.RESET}")
            print(f"{Config.BRIGHT_CYAN}║{Config.RESET} {Config.BOLD}RESPOSTA{Config.RESET}")
            print(f"{Config.BRIGHT_CYAN}╚{'═'*48}╝{Config.RESET}\n")
            print(result)
            
            self.brain.add_interaction("prompt", topic, result)
            self.logger.log("prompt_generation", topic, result)
            
            print(f"\n{Config.GREEN}💾 Salvo{Config.RESET}")
        else:
            print(f"{Config.RED}❌ Falha{Config.RESET}")
    
    def create_mindmap(self):
        """Criar mapa mental"""
        print(f"\n{Config.CYAN}{'─'*50}")
        print(f"  🧠 CRIADOR DE MAPA MENTAL")
        print(f"{'─'*50}{Config.RESET}\n")
        
        topic = input(f"{Config.YELLOW}🎯 Tópico:{Config.RESET} ").strip()
        
        if not topic:
            print(f"{Config.RED}❌ Vazio{Config.RESET}")
            return
        
        template = f"""Create a detailed mind map for: {topic}

CENTRAL CONCEPT: {topic}

MAIN BRANCHES (5-7):
- Branch 1: [Category]
  - Sub 1.1: [Detail]
  - Sub 1.2: [Detail]

- Branch 2: [Category]
  - Sub 2.1: [Detail]

[Continue...]

KEY CONNECTIONS:
- Relationships between branches

APPLICATIONS:
- Real-world uses

Format as clear indented text."""
        
        result = self.client.generate(template, max_tokens=1000, temperature=0.8)
        
        if result:
            print(f"\n{Config.BRIGHT_CYAN}╔{'═'*48}╗{Config.RESET}")
            print(f"{Config.BRIGHT_CYAN}║{Config.RESET} {Config.BOLD}MAPA MENTAL{Config.RESET}")
            print(f"{Config.BRIGHT_CYAN}╚{'═'*48}╝{Config.RESET}\n")
            print(result)
            
            self.brain.add_interaction("mindmap", topic, result)
            self.logger.log("mindmap_creation", topic, result)
            
            print(f"\n{Config.GREEN}💾 Salvo{Config.RESET}")
        else:
            print(f"{Config.RED}❌ Falha{Config.RESET}")
    
    def show_stats(self):
        """Estatísticas"""
        print(f"\n{Config.CYAN}{'─'*50}")
        print(f"  📈 ESTATÍSTICAS")
        print(f"{'─'*50}{Config.RESET}\n")
        
        stats = self.brain.get_stats()
        
        print(f"{Config.BOLD}Uso:{Config.RESET}")
        print(f"  • Total: {Config.BRIGHT_CYAN}{stats['total']}{Config.RESET}")
        print(f"  • Prompts: {Config.BRIGHT_GREEN}{stats['prompts']}{Config.RESET}")
        print(f"  • Mapas: {Config.YELLOW}{stats['mindmaps']}{Config.RESET}")
        print(f"  • Sessões: {Config.CYAN}{stats['sessions']}{Config.RESET}")
        
        print(f"\n{Config.BOLD}Sistema:{Config.RESET}")
        print(f"  • Modelo: {Config.CYAN}Qwen 2.5 7B{Config.RESET}")
        print(f"  • Versão: {Config.BRIGHT_CYAN}2.0.0{Config.RESET}")
        
        if self.brain.data["interactions"]:
            print(f"\n{Config.BOLD}Últimas 3:{Config.RESET}")
            for i in self.brain.data["interactions"][-3:]:
                ts = i["timestamp"].split("T")[1].split(".")[0]
                print(f"  [{ts}] {i['type']}: {i['prompt'][:40]}...")
    
    def show_about(self):
        """Sobre"""
        print(f"\n{Config.CYAN}{'─'*50}")
        print(f"  📚 SOBRE")
        print(f"{'─'*50}{Config.RESET}\n")
        
        print(f"{Config.BOLD}Genyx AI v2.0.0{Config.RESET}")
        print(f"\n{Config.BRIGHT_CYAN}Características:{Config.RESET}")
        print(f"  • Requests direto (sem deps pesadas)")
        print(f"  • Qwen 2.5 7B Instruct")
        print(f"  • Logging para treinamento")
        print(f"  • Persistência JSON")
        print(f"  • Termux otimizado")
        
        print(f"\n{Config.BRIGHT_GREEN}Arquivos:{Config.RESET}")
        print(f"  • brain_history.json")
        print(f"  • learning_log.txt")
        
        print(f"\n{Config.YELLOW}Config:{Config.RESET}")
        print(f"  • Token: HF_TOKEN")
        print(f"  • Timeout: {Config.API_TIMEOUT}s")
    
    def run(self):
        """Loop principal"""
        os.system('clear')
        self.ui.print_banner()
        
        print(f"{Config.GREEN}✅ Token configurado{Config.RESET}")
        print(f"{Config.CYAN}🔗 Qwen 2.5 7B Instruct{Config.RESET}")
        
        while True:
            stats = self.brain.get_stats()
            self.ui.print_menu(stats)
            
            try:
                choice = input(f"{Config.BOLD}➤ Opção:{Config.RESET} ").strip()
                
                if choice == '1':
                    self.generate_prompt()
                elif choice == '2':
                    self.create_mindmap()
                elif choice == '3':
                    self.show_stats()
                elif choice == '4':
                    self.show_about()
                elif choice == '0':
                    self.logger.log_event("SYSTEM_EXIT")
                    print(f"\n{Config.BRIGHT_GREEN}👋 Até logo!{Config.RESET}\n")
                    break
                else:
                    print(f"{Config.RED}❌ Inválida{Config.RESET}")
                
                input(f"\n{Config.YELLOW}⏎ Enter...{Config.RESET}")
            
            except KeyboardInterrupt:
                self.logger.log_event("INTERRUPT")
                print(f"\n{Config.YELLOW}⚠️  Interrompido{Config.RESET}")
                break
            except Exception as e:
                print(f"{Config.RED}❌ Erro: {e}{Config.RESET}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        app = GenyxAI()
        app.run()
    except Exception as e:
        print(f"{Config.RED}❌ Fatal: {e}{Config.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
