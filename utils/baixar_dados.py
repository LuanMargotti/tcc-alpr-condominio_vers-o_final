from roboflow import Roboflow

rf = Roboflow(api_key="2DCl4425peaRVjOakYeu")

project = rf.workspace("teste-yfcdp").project("placas-brasil-no2nm")

version = project.version(2)

dataset = version.download("yolov8")

print(f"Dataset baixado com sucesso em: {dataset.location}")