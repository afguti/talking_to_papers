
def separate(text: str, host="Host", guest="Guest") -> list:
    podcast_host_lines = []
    guest_lines = []

    lines = text.strip().split('\n')
    current_speaker = None
    current_dialogue = ""

    for line in lines:
        if line.strip() == "":
            continue  # Skip empty lines
        if ":" in line:
            if current_speaker is not None:
                if current_speaker == host:
                    podcast_host_lines.append(current_dialogue)
                elif current_speaker == guest:
                    guest_lines.append(current_dialogue)
            current_speaker, current_dialogue = line.split(': ', 1)
        else:
            current_dialogue += " " + line.strip()

    # Add the last speaker's dialogue
    if current_speaker is not None:
        if current_speaker == host:
            podcast_host_lines.append(current_dialogue)
        elif current_speaker == guest:
            guest_lines.append(current_dialogue)

    return podcast_host_lines, guest_lines