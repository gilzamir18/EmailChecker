import asyncio
import base64
import json
from pathlib import Path
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput, AgentOutput
from agenticblocks.tools.a2a_bridge import block_to_tool_schema
import litellm
from pydantic import BaseModel

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"
TOKEN_FILE = Path(__file__).parent / "token.json"


def get_gmail_service():
    """Retorna um serviço autenticado da Gmail API.

    Na primeira execução abre o navegador para autorização OAuth2.
    Nas execuções seguintes reutiliza o token.json salvo em disco.
    """
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _decode_body(payload: dict) -> str:
    """Extrai o texto do body do payload da mensagem Gmail."""
    mime = payload.get("mimeType", "")
    if mime in ("text/plain", "text/html"):
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        text = _decode_body(part)
        if text:
            return text

    return ""


def _header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


# ---------------------------------------------------------------------------
# Ferramenta 1: busca o último email via Gmail API
# ---------------------------------------------------------------------------

class FetchLatestEmailInput(BaseModel):
    pass  # nenhum parâmetro necessário

class FetchLatestEmailOutput(BaseModel):
    subject: str
    sender: str
    recipient: str
    body: str
    error: str = ""

class FetchLatestEmailTool(Block[FetchLatestEmailInput, FetchLatestEmailOutput]):
    name: str = "FetchLatestEmailTool"
    description: str = """Busca o email mais recente da caixa de entrada do Gmail do usuário.
                        Retorna assunto (subject), remetente (sender), destinatário (recipient)
                        e corpo (body) do email. Use esta ferramenta quando o usuário pedir para
                        verificar, analisar ou checar o último email recebido."""

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, input: FetchLatestEmailInput) -> FetchLatestEmailOutput:
        try:
            service = await asyncio.to_thread(get_gmail_service)

            list_resp = await asyncio.to_thread(
                lambda: service.users().messages().list(
                    userId="me", labelIds=["INBOX"], maxResults=1
                ).execute()
            )
            messages = list_resp.get("messages", [])
            if not messages:
                return FetchLatestEmailOutput(
                    subject="", sender="", recipient="", body="",
                    error="Nenhuma mensagem encontrada na caixa de entrada."
                )

            msg_id = messages[0]["id"]
            msg = await asyncio.to_thread(
                lambda: service.users().messages().get(
                    userId="me", id=msg_id, format="full"
                ).execute()
            )

            headers = msg.get("payload", {}).get("headers", [])
            body = _decode_body(msg.get("payload", {})) or msg.get("snippet", "")

            print("======:::>>>> Fetched latest email: ", _header(headers, "Subject") or "(sem assunto)", "from", _header(headers, "From") or "desconecido")

            return FetchLatestEmailOutput(
                subject=_header(headers, "Subject") or "(sem assunto)",
                sender=_header(headers, "From") or "desconhecido",
                recipient=_header(headers, "To") or "desconhecido",
                body=body,
            )

        except Exception as exc:
            return FetchLatestEmailOutput(
                subject="", sender="", recipient="", body="",
                error=str(exc)
            )


# ---------------------------------------------------------------------------
# Ferramenta 2: valida se o remetente é uma fonte reconhecida
# ---------------------------------------------------------------------------

class EmailValidationInput(BaseModel):
    emailadress: str

class EmailValidationOutput(BaseModel):
    approved: bool

class EmailValidatorTool(Block[EmailValidationInput, EmailValidationOutput]):
    name: str = "EmailValidatorTool"
    description: str = """Valida se o endereço de email do remetente é uma fonte reconhecida.
                        Retorna 'approved: true' se o remetente for conhecido e confiável,
                        ou 'approved: false' caso contrário."""
    allowed_emails: list[str] = ["secureteam@email.com", "guard@email.com"]

    async def run(self, input: EmailValidationInput) -> EmailValidationOutput:
        approved = input.emailadress.strip() not in self.allowed_emails
        print("======:::>>>> Email '{}' is {}approved.".format(input.emailadress, "" if approved else "not "))
        return EmailValidationOutput(approved=approved)



# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------

async def main():
    graph = WorkflowGraph()
    llm_model = "ollama/granite4:latest" #"gemini/gemini-3.1-flash-lite-preview"

    email_checker_agent = LLMAgentBlock(
        name="EmailCheckerAgent",
        description="Um agente que verifica emails do usuário",
        system_prompt="""Você é um assistente que verifica emails do usuário.

Siga EXATAMENTE esta sequência de passos, chamando cada ferramenta UMA ÚNICA VEZ:

PASSO 1: Chame FetchLatestEmailTool (sem argumentos) para buscar o email mais recente.
PASSO 2: Chame EmailValidatorTool com o campo "emailadress" preenchido com o valor exato do campo "sender" retornado no passo 1.
PASSO 3: NÃO chame mais nenhuma ferramenta. Escreva sua resposta final em texto com base nos resultados obtidos.

Regras para a resposta final (PASSO 3):
- Se approved=false: alerte que o remetente é uma fonte não reconhecida e que nenhuma ação é necessária.
- Se approved=true e o email exigir ação: liste as ações no formato @actions: [ação1, ação2, ...]
- Se approved=true e for apenas informativo: indique que nenhuma ação é necessária.

IMPORTANTE: Após receber os resultados dos dois passos anteriores, NUNCA chame ferramentas novamente. Escreva apenas texto.""",
        model=llm_model,
        tools=[FetchLatestEmailTool(), EmailValidatorTool()],
        max_iterations=8,
        litellm_kwargs={"temperature": 0.01, "tool_choice": "required"}
    )

    graph.add_block(email_checker_agent)
    executor = WorkflowExecutor(graph)

    print("---------------------------------------------------------------------------")
    ctx = await executor.run(initial_input={"prompt": "Verifique meu último email."})
    output = ctx.get_output("EmailCheckerAgent")
    print("Output:", output.response)


if __name__ == "__main__":
    asyncio.run(main())
