# MyShell

## Overview

`MyShell` is a simple Unix-like shell implemented in C that supports a variety of shell functionalities including:

- Command execution
- Input and output redirection (`>`, `>>`, `<`)
- Command piping (`|`)
- Background execution (`&`)
- Change directory (`cd`)
- Command history (optional feature)
- Prompt with clear screen

This shell imitates basic behavior similar to bash or sh, and executes commands by interacting with the system's process control mechanisms (`fork`, `exec`, `dup`, `pipe`).

## Features

1. **Command Prompt**: Displays a simple `$mysh>` prompt to the user.
2. **Input/Output Redirection**: Supports `>` for overwriting and `>>` for appending output to files, as well as `<` for input redirection from files.
3. **Piping**: Allows the use of pipes (`|`) to connect multiple commands, where the output of one command becomes the input for another.
4. **Background Processes**: Supports running processes in the background using the `&` symbol.
5. **Change Directory (`cd`)**: Implements the built-in `cd` command for changing directories.
6. **Clear Screen**: Clears the terminal screen with the `clear` command.
7. **Custom Path**: Supports command execution using a custom path (`/bin/`).
8. **Error Handling**: Handles errors like invalid command execution, failed redirections, or unsuccessful forks.

## Installation

To compile and run `MyShell`, follow these steps:

1. Clone or download the source code.
2. Open a terminal and navigate to the directory where the source code is located.
3. Compile the code using `gcc`:

    ```bash
    gcc -o mysh mysh.c
    ```

4. Run the shell:

    ```bash
    ./mysh
    ```

## Commands

### Basic Usage

- **Execute a Command**: Type a command and hit enter, for example:

    ```bash
    $mysh> ls
    ```

- **Change Directory**: Use `cd` to change the current working directory:

    ```bash
    $mysh> cd /home/user/documents
    ```

- **Exit Shell**: Type `exit` to quit the shell:

    ```bash
    $mysh> exit
    ```

- **Clear Screen**: Type `clear` to clear the terminal screen:

    ```bash
    $mysh> clear
    ```

### Redirection

- **Redirect Output**: Use `>` to overwrite the contents of a file with the output of a command:

    ```bash
    $mysh> echo "Hello World" > output.txt
    ```

- **Append Output**: Use `>>` to append the output of a command to an existing file:

    ```bash
    $mysh> echo "New line" >> output.txt
    ```

- **Redirect Input**: Use `<` to take input from a file:

    ```bash
    $mysh> sort < input.txt
    ```

### Piping

You can pipe the output of one command into another:

```bash
$mysh> cat file.txt | grep "pattern"
