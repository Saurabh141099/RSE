from rest_framework import fields, serializers

class SearchSerializers(serializers.Serializer):
    image = serializers.FileField()
