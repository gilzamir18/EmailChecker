import asyncio
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import AgentInput
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from pydantic import BaseModel

class EmailValidationInput(BaseModel):
    emailadress: str

class EmailValidationOutput(BaseModel):
    approved: bool

class EmailValidatorTool(Block[EmailValidationInput, EmailValidationOutput]):
    name: str = "EmailValidatorTool"
    description: str = """Uma ferramenta que valida se um email é apenas 
                        informativo ou se exige uma resposta. Retorna 'approved: true' se 
                        o email exigir uma resposta e 'approved: false' caso contrário."""
    allowed_emails: list[str] = ["secureteam@email.com", "guard@email.com"]

    async def run(self, input: EmailValidationInput) -> EmailValidationOutput:
        if input.emailadress.strip() in self.allowed_emails:
            #print(f"======:::>>>> Email '{input.emailadress}' is approved.")
            return EmailValidationOutput(approved=True)
        else:
            #print(f"======:::>>>> Email '{input.emailadress}' is not approved.")
            return EmailValidationOutput(approved=False)

async def main():
    graph = WorkflowGraph()
    llm_model = "ollama/granite4:latest"
    email_checker_agent = LLMAgentBlock(
            name="EmailCheckerAgent",
            description="Um agente que verifica se um email é apenas informativo ou se exige uma resposta",
            system_prompt="""Você é um agente que verifica se um email é apenas informativo
                         ou se exige uma resposta. Se o email exigir uma resposta, sugira 
                         ações a serem tomadas. A lista de ações deve ser clara e concisa.
                         Se o email for apenas informativo, indique que nenhuma ação é necessária.
                         Coloque a lista de ações no formato de uma lista numerada usando
                         a forma @actions: [ação1, ação2, ...]. 
                         Use a ferramenta EmailValidatorTool para validar o email do remetente.
                         O campo "approved" da resposta da ferramenta indicará se o email é aprovado ou não.
                         Se o email não for aprovado pela ferramenta, indique que nenhuma ação é necessária.
                         Mas alerte que o email é de uma fonte não reconhecida!""",
            model=llm_model,
            tools=[EmailValidatorTool()],
            max_iterations=5,
            litellm_kwargs={"temperature": 0.1}
        )
    
    graph.add_block(email_checker_agent)

    emails = ["segurança@email.com", "secureteam@email.com"]

    executor = WorkflowExecutor(graph)
    for email in emails:
        print("---------------------------------------------------------------------------")
        print("\n\nTrying send email: ", email)
        ctx = await executor.run(initial_input={"prompt": 
                                                f"""O email é: 'Assunto: Urgente! Atualize sua senha. 
                                                De: {email}
                                                Para: usuario@email.com
                                                Corpo: Prezado usuário, detectamos uma tentativa 
                                                de acesso suspeita em sua conta."""})
        output = ctx.get_output("EmailCheckerAgent")
        print("Output: ", output.response)

if __name__ == "__main__":
    asyncio.run(main())