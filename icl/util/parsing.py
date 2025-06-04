def parse_config_args(config_args):
    config_override_dict = {}
    for i in range(0, len(config_args), 2):
        key = config_args[i].lstrip('--')
        value = config_args[i+1]
        
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        elif value.lower() in {"true", "false"}:
            value = value.lower() == "true"
            
        config_override_dict[key] = value
    return config_override_dict