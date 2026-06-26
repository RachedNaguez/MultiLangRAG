"""Multilingual document store."""

from __future__ import annotations

from langchain_core.documents import Document

DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "The company policy allows for up to 20 days of paid time off per year. "
            "Employees must request time off at least two weeks in advance."
        ),
        metadata={"language": "English", "source": "HR_Policy_EN"},
    ),
    Document(
        page_content=(
            "La politique de l'entreprise accorde jusqu'a 20 jours de conges payes "
            "par an. Les employes doivent en faire la demande au moins deux semaines "
            "a l'avance."
        ),
        metadata={"language": "French", "source": "HR_Policy_FR"},
    ),
    Document(
        page_content=(
            "El soporte tecnico esta disponible 24/7. Para problemas urgentes, llame "
            "al numero de emergencia en lugar de enviar un correo electronico."
        ),
        metadata={"language": "Spanish", "source": "IT_Policy_ES"},
    ),
]
