# utils.py

import ast


def is_valid_hex(value):
    """
    Verifica se o valor é um hexadecimal válido.

    Parâmetros:
    - value (str): O valor a ser verificado.

    Retorna:
    - bool: True se válido, False caso contrário.
    """
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def converter_string_para_lista(string_valor):
    """
    Converte uma string representando uma lista em uma lista de floats.

    Parâmetros:
    - string_valor (str): A string a ser convertida.

    Retorna:
    - list: Lista de floats ou uma lista vazia em caso de erro.
    """
    try:
        return ast.literal_eval(string_valor)
    except (ValueError, SyntaxError) as e:
        print(f"Erro ao converter string para lista: {e}")
        return []
