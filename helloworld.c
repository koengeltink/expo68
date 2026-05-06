#include "xparameters.h"
#include "xuartps.h"
#include <string.h>
#include <stdlib.h>

#define UART_BAUDRATE 9600

XUartPs UartPs;

int main(void)
{
    u8 RecvBuffer[32];
    u8 SendBuffer[32];
    u8 RecvByte;
    int BytesReceived;

    // Init UART0
    XUartPs_Config *Config = XUartPs_LookupConfig(XPAR_XUARTPS_0_DEVICE_ID);
    XUartPs_CfgInitialize(&UartPs, Config, Config->BaseAddress);
    XUartPs_SetBaudRate(&UartPs, UART_BAUDRATE);
    XUartPs_SetOperMode(&UartPs, XUARTPS_OPER_MODE_NORMAL);

    while (1)
    {
        BytesReceived = 0;
        memset(RecvBuffer, 0, sizeof(RecvBuffer));

        // Read until newline
        while (BytesReceived < 31)
        {
            while (XUartPs_Recv(&UartPs, &RecvByte, 1) == 0);
            if (RecvByte == '\n') break;
            RecvBuffer[BytesReceived++] = RecvByte;
        }

        if (BytesReceived > 0)
        {
            int sliderValue = atoi((char*)RecvBuffer);
            int len = snprintf((char*)SendBuffer, sizeof(SendBuffer), "ACK:%d\n", sliderValue);
            XUartPs_Send(&UartPs, SendBuffer, len);
            while (XUartPs_IsSending(&UartPs));
        }
    }
    return 0;
}
