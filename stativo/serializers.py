from rest_framework import serializers


class NaturalLanguageQuerySerializer(serializers.Serializer):
    query = serializers.CharField(
        max_length=4096,
        allow_blank=False,
        help_text="Natural language query for the Firestore AI agent"
    )


class NaturalLanguageResponseSerializer(serializers.Serializer):
    response = serializers.CharField(
        help_text="Natural language response from the Firestore AI agent"
    )
