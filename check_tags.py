from PIL import ExifTags

search_fields = ['ResolutionUnit', 'YResolution', 'XResolution', 'YCbCrPositioning', 'Make', 'Model']
for f in search_fields:
    ids = [id for id, name in ExifTags.TAGS.items() if name == f]
    print(f"{f}: {ids}")
