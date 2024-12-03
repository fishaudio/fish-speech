import html


def build_html_error_message(error : Exception) -> str:
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """