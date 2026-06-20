export const defaultWsUri= "ws://127.0.0.1:8000/";

export function startWebsocket(wsUri: string = defaultWsUri) {
   return new WebSocket(wsUri)
}