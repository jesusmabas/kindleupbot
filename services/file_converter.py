# kindleupbot/services/file_converter.py
import asyncio
import logging
import pypandoc
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

async def convert_markdown_to_docx(md_path: Path, title: str) -> Tuple[Optional[Path], Optional[str]]:
    docx_path = md_path.with_suffix('.docx')
    base_title = Path(title).stem
    metadata_args = [f'--metadata=title:{base_title}']
    try:
        await asyncio.to_thread(
            pypandoc.convert_file,
            str(md_path),
            'docx',
            outputfile=str(docx_path),
            extra_args=['--standalone'] + metadata_args
        )
        logger.info(f"Archivo Markdown convertido exitosamente a DOCX en: {docx_path}")
        return docx_path, None
    except Exception as e:
        logger.error(f"Error al convertir Markdown a DOCX con Pandoc: {e}", exc_info=True)
        return None, f"Error de Pandoc: {str(e)}"