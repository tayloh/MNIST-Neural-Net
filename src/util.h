#include <stdio.h>

// Sigh...

#ifdef _WIN32
#include <windows.h>
#endif

void set_console_csr_xy(int x, int y)
{
#ifdef _WIN32
    COORD coord;
    coord.X = x;
    coord.Y = y;
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
#endif
}

int get_current_cursor_row()
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.dwCursorPosition.Y;
#endif
}

void draw_progress_bar(int percentage, int row)
{
#ifdef _WIN32
    int width = 50;
    int pos = (percentage * width) / 100;

    set_console_csr_xy(0, row); // Move to start of the row
    printf("[");
    for (int i = 0; i < width; ++i)
    {
        if (i <= pos)
            printf("=");
        else
            printf(" ");
    }
    printf("] %3d%%", percentage);
    fflush(stdout);
#endif
}

// // Usage:
//     int start_row = get_current_cursor_row(); // Get row just after "Starting operation..."

//     for (int i = 0; i <= 100; ++i)
//     {
//         draw_progress_bar(i, start_row);
//         Sleep(50); // Simulate work
//     }

//     set_console_csr_xy(0, start_row + 1); // Move below progress bar