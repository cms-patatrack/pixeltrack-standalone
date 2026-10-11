from std.pathlib import Path

from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.bin.Plugins import ed_path, es_path, es_produce, make_es


def main() raises:
    var evt = EventSetup()

    var esp_names = es_path()
    for i in range(len(esp_names)):
        var esp = make_es(esp_names[i], Path("data"))
        es_produce(esp, evt)

    var ed_names = ed_path(False)
    for i in range(len(ed_names)):
        print(ed_names[i])
