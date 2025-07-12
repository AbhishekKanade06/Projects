#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <fcntl.h>

void prompt()
{
    static int first = 1;
    if (first)
    {
        system("clear");
        first = 0;
    }
    write(1, "$mysh>", 6);
}

void read_command(char *command, char **args)
{
    char *token;
    token = strtok(command, " \n");
    int i = 0;
    while (token != NULL)
    {
        args[i++] = token;
        token = strtok(NULL, " \n");
    }
    args[i] = NULL;
}

int output_redirect(char **args, int out_red)
{
    for (int i = 0; args[i] != NULL; i++)
    {
        if (strcmp(args[i], ">") == 0 || strcmp(args[i], ">>") == 0)
        {
            int save_stdo = dup(1);
            int redir;
            if (out_red == 2)
            {
                redir = open(args[i + 1], O_WRONLY | O_CREAT | O_APPEND, 0644);
            }
            else
            {
                redir = open(args[i + 1], O_CREAT | O_TRUNC | O_WRONLY, 0644);
            }
            dup2(redir, 1);
            close(redir);
            args[i] = NULL; // NULL terminate to break command at redirection
            return save_stdo;
        }
    }
    return -1; // no redirection found
}

int input_redirect(char **args)
{
    for (int i = 0; args[i] != NULL; i++)
    {
        if (strcmp(args[i], "<") == 0)
        {
            int save_stdi = dup(0);
            int redir = open(args[i + 1], O_RDONLY);
            dup2(redir, 0);
            close(redir);
            args[i] = NULL; // NULL terminate to break command at redirection
            return save_stdi;
        }
    }
    return -1; // no redirection found
}

void execute_command(char **args[], char path[], int in_red, int out_red, int save_sdtio, int multi)
{
    pid_t pid = fork();
    if (pid == 0)
    {
        strcat(path, args[0]);
        if (open(args[0], O_RDONLY) != -1)
        {
            execl(args[0], args[0], NULL);
        }
        else if(open(path,O_RDONLY)!=-1)
        { 
          
            execl(path, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], NULL);

        }
        else {
             execlp(args[0],args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], NULL);
        }
        perror("execlp");
        exit(1);
    }

    else if (pid < 0)
    {
        perror("fork");
    }

    else
    {
        if (out_red==1||out_red==2)
        {
            close(1);
            dup2(save_sdtio, 1);
            close(save_sdtio);
        }
        if (in_red)
        {
            close(0);
            dup2(save_sdtio, 0);
            close(save_sdtio);
        }
        if (!multi)
        {
            wait(NULL);
        }
    }
}
void handle_pipe(char **args, int pip_idx)
{
    char *args1[50];
    char *args2[50];
    int idx = 0;
    for (int i = 0; i < pip_idx; i++)
    {
        args1[i] = args[i];
    }
    args1[pip_idx] = NULL;

    for (int i = pip_idx + 1; args[i] != NULL; i++)
    {
        args2[idx++] = args[i];
    }
    args2[idx] = NULL;

    int pf[2];
    pipe(pf);
    pid_t pid1 = fork();
    if (pid1 == 0)
    {
        close(1);
        dup2(pf[1], 1);
        close(pf[0]);
        close(pf[1]);
        execvp(args1[0], args1);
        perror("execvp");
        exit(1);
    }
    else if (pid1 < 0)
    {
        perror("fork");
    }

    pid_t pid2 = fork();
    if (pid2 == 0)
    {
        close(0);
        dup2(pf[0], 0);
        close(pf[1]);
        close(pf[0]);
        execvp(args2[0], args2);
        perror("execvp");
        exit(1);
    }
    else if (pid2 < 0)
    {
        perror("fork");
    }

    close(pf[0]);
    close(pf[1]);
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);
}

int main()
{
    char *command = NULL;
    char *args[100];
    size_t bufsize = 0;

    while (1)
    {
        int multi = 0;
        int pipe_f = 0, pip_idx = 0;
        char path[] = "/bin/";
        int out_red = 0;
        int in_red = 0;
        int save_stdio = -1;

        prompt();
        getline(&command, &bufsize, stdin);

        if (strcmp(command, "exit\n") == 0)
        {
            free(command);
            exit(0);
        }
        else if (strcmp(command, "clear\n") == 0)
        {
            system("clear");
            continue;
        }

        read_command(command, args);
        if(strcmp(args[0],"cd")==0){
            if(chdir(args[1])!=0){
                perror("chdir");
            }
            continue;
        }

        for (int i = 0; args[i] != NULL; i++)
        {
            if (strcmp(args[i], "|") == 0)
            {
                pipe_f = 1;
                pip_idx = i;
                break;
            }
        }

        if (pipe_f)
        {
            handle_pipe(args, pip_idx);
        }
        else
        {
            for (int i = 0; args[i] != NULL; i++)
            {
                if (strcmp(args[i], "&") == 0)
                {
                    multi = 1;
                    args[i] = NULL;
                    break;
                }
            }

            for (int i = 0; args[i] != NULL; i++)
            {
                if (strcmp(args[i], ">") == 0)
                {
                    out_red = 1;
                    save_stdio = output_redirect(args, out_red);
                    break;
                }
                if (strcmp(args[i], ">>") == 0)
                {
                    out_red = 2;
                    save_stdio = output_redirect(args, out_red);
                    break;
                }
            }

            for (int i = 0; args[i] != NULL; i++)
            {
                if (strcmp(args[i], "<") == 0)
                {
                    save_stdio = input_redirect(args);
                    in_red = 1;
                    break;
                }
            }

            execute_command(args, path, in_red, out_red, save_stdio, multi);
        }

        free(command);
        command = NULL; // reset command for next getline
    }

    return 0;
}
