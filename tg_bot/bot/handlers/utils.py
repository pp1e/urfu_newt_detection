def escape_md(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
            .replace("*", "\\*")
            .replace("_", "\\_")
            .replace("`", "\\`")
            .replace("[", "\\[")
    )
