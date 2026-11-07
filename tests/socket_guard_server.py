import os
import socket

LOG = os.environ["MEETSCRIBE_SOCKET_LOG"]
LOCAL = {"127.0.0.1", "::1", "localhost", "", None}
real_getaddrinfo = socket.getaddrinfo


def note(kind, target):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{kind} {target!r}\n")


def is_local(address):
    return not isinstance(address, tuple) or address[0] in LOCAL


class GuardedSocket(socket.socket):
    def connect(self, address):
        if not is_local(address):
            note("connect", address)
            raise OSError("Netzzugriff gesperrt")
        return super().connect(address)

    def connect_ex(self, address):
        if not is_local(address):
            note("connect_ex", address)
            raise OSError("Netzzugriff gesperrt")
        return super().connect_ex(address)

    def sendto(self, data, *args):
        note("sendto", args[-1])
        raise OSError("Netzzugriff gesperrt")


def getaddrinfo(host, *args, **kwargs):
    if host not in LOCAL:
        note("getaddrinfo", host)
        raise socket.gaierror("Netzzugriff gesperrt")
    return real_getaddrinfo(host, *args, **kwargs)


socket.socket = GuardedSocket
socket.getaddrinfo = getaddrinfo

if __name__ == "__main__":
    from meetscribe.mcp_server import main
    main()
