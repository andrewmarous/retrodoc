import argparse, ast, os
from _ast import FunctionDef, Import
from redbaron import RedBaron

from openai import OpenAI


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    """
    Main function that parses command line arguments and calls the document function.
    
    Parameters:
    None
    
    Returns:
    None
    """
    parser = argparse.ArgumentParser(description='retrodoc CLI v1.0')
    parser.add_argument('filename', type=str, help='Name of the file to be documented')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite current documentation in file')
    parser.add_argument('--lang', '-l', type=str, help='Optional language annotation for clarity')

    args = parser.parse_args()

    document(args)


def document(args):
    """
    Generates specifications for a given file and writes them to the file.
    
    Parameters:
    args (object): An object containing the arguments for the documentation process.
    
    Returns:
    None
    """
    try:
        filepath = os.path.join(os.getcwd(), args.filename)
        tokens = tokenize_code(filepath)
        comments = generate_comments(tokens)
        print('Specifications generated. Writing to ' + args.filename + '...')
        write_comments(comments, filepath, args.overwrite)
        print(args.filename + ' successfully documented.')

    except FileNotFoundError:
        print("Error: the file is either empty or does not exist. Please check that your filepath/filename are valid. ")


def tokenize_code(filepath):
    """
    Tokenizes the code from a file, separating functions and import statements.
    
    Parameters:
    filepath (str): The path to the file containing the code.
    
    Returns:
    dict: A dictionary containing two keys: 'imports' for the list of import statements and 'funcs' for the list of function definitions.
    """
    if os.path.getsize(filepath) == 0:
        return []

    with open(filepath, 'r', encoding = 'utf8') as f:
        source = f.read()

    funcs_ast = ast.parse(source, filepath)
    statements = dict([('imports', []), ('funcs', [])])

    class MethodVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: FunctionDef):
            """
            Generates comments for the provided tokens using the OpenAI GPT-3.5 Turbo model.
            
            Parameters:
            tokens (dict): A dictionary containing a key 'funcs' with a list of tokens to generate comments for.
            
            Returns:
            list: A list of generated comments for each token in the 'funcs' list.
            """
            statements.get('funcs').append(ast.unparse(node))

        def visit_Import(self, node: Import):
            """
            Writes comments from a list to the specified file's functions as docstrings.
            
            Parameters:
            comments (list): A list of comments to add as docstrings.
            filepath (str): The path to the file containing the source code.
            overwrite (bool): A flag indicating whether existing docstrings should be overwritten.
            
            Returns:
            None
            """
            statements.get('imports').append(ast.unparse(node))

    MethodVisitor().visit(funcs_ast)

    return statements


def generate_comments(tokens: dict):
    comments = []
    client = OpenAI()
    for token in tokens.get('funcs'):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system', 'content': GPT_SYSTEM_PROMPT_PY},
                {'role': 'user', 'content': token}
            ]
        )
        comments.append(completion.choices[0].message)
    return comments


def write_comments(comments, filepath, overwrite):
    with open(filepath, 'r') as file:
        source_code = file.read()

    # Parse code
    red = RedBaron(source_code)

    # Iterator for docstrings to be inserted
    comments_iter = iter(comments)

    # Iterate over all function definitions and insert docstrings
    for function_node in red.find_all('def'):
        try:
            comment = next(comments_iter)

            # create comment node
            indented_docstring_lines = [function_node.indentation + line if i != 0 else line
                                        for i, line in enumerate(comment.content.split('\n'))]

            indented_docstring = "\n".join(indented_docstring_lines)
            # Check if the function already has a docstring

            if function_node.value and not (function_node.value[0].type == 'string' or
                                            function_node.value[0].type == 'raw_string'):
                function_node.value.insert(0, RedBaron("placeholder"))
                function_node.value[0] = indented_docstring
            elif overwrite:
                function_node.value[0] = indented_docstring

        except StopIteration:
            break

    with open(filepath, 'w') as file:
        file.write(red.dumps())








GPT_SYSTEM_PROMPT_PY = '''You are a documenter for Python methods: when you are given a method, you reply with the 
proper documentation for the method in the Python standard. Return only the specification docstring, without the 
function declaration. The docstring must be returned in a very strict format, with only three double-quotation marks at 
the beginning and end of the docstring. The first set of quotes must be immediately followed by a newline character, 
and the last set of quotes must be immediately preceded by a newline character. The docstring must follow exactly this
format: 
"""
Single-sentence method description.

Parameters:
Describe each parameter on a separate line

Returns:
Describe the returned value/object. Say 'None' if the method returns nothing.
"""

Here is an example, for a function that implements the merge sort algorithm:
"""
Sorts the elements of the input array using the merge sort algorithm.

Parameters:
arr (list): The input list to be sorted.

Returns:
None
"""

Ignore any specification that already exists in the method. You are only to analyze the code and return a specification
based on that.
'''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
