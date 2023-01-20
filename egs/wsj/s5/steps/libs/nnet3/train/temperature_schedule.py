def get_temperature_edit_string(temperature_init, temperature_final, data_fraction, iter_):
    
    if (temperature_init is None) or  (temperature_final is None):
        return
    
    temperature = temperature_init*(temperature_final/temperature_init)**data_fraction
    
    edit_config_lines = []
    temperature_info = []
    
    edit_config_lines.append(
        "set-temperature temperature={0}".format(
            temperature))
    temperature_info.append("temperature={0}".format(
        temperature))
    
    return ("""nnet3-copy --edits='{edits}' - - |""".format(
        edits=";".join(edit_config_lines)))
