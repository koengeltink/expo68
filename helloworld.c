#include "xparameters.h"
#include "xuartps.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define UART_BAUDRATE 9600

XUartPs UartPs;

// Reliable send function (IMPORTANT FIX)
void uart_send_string(XUartPs *Uart, const char *str)
{
    int len = strlen(str);
    int totalSent = 0;

    while (totalSent < len)
    {
        int sent = XUartPs_Send(Uart, (u8*)str + totalSent, len - totalSent);
        totalSent += sent;
    }

    // Wait until fully transmitted
    while (XUartPs_IsSending(Uart));
}

int main(void)
{
    u8 RecvBuffer[32];
    u8 RecvByte;
    int BytesReceived;

    // Init UART0 (PROG-UART USB)
    XUartPs_Config *Config = XUartPs_LookupConfig(XPAR_XUARTPS_0_DEVICE_ID);
    XUartPs_CfgInitialize(&UartPs, Config, Config->BaseAddress);
    XUartPs_SetBaudRate(&UartPs, UART_BAUDRATE);
    XUartPs_SetOperMode(&UartPs, XUARTPS_OPER_MODE_NORMAL);

    // 🔍 Startup test (you should see this on the Pi)
    uart_send_string(&UartPs, "BOOT OK\n");

    while (1)
    {
        BytesReceived = 0;
        memset(RecvBuffer, 0, sizeof(RecvBuffer));

        // Read until newline
        while (BytesReceived < 31)
        {
            while (XUartPs_Recv(&UartPs, &RecvByte, 1) == 0);

            if (RecvByte == '\n')
                break;

            RecvBuffer[BytesReceived++] = RecvByte;
        }

        if (BytesReceived > 0)
        {
            int sliderValue = atoi((char*)RecvBuffer);

            char SendBuffer[32];
            snprintf(SendBuffer, sizeof(SendBuffer), "ACK:%d\n", sliderValue);

            // ✅ Use fixed send function
            uart_send_string(&UartPs, SendBuffer);
        }
    }

    return 0;
}