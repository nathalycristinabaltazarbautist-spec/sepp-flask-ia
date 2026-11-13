from ultralytics import YOLO

# Cargar el archivo de pesos descargado
model = YOLO('Epp_RN3.pt')

# Imprimir el diccionario exacto de clases detectables
print("--- CLASES DETECTADAS EN EL MODELO ---")
print(model.names)