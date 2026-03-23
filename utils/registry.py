class Registry(dict):
    def register(self, name):
        def decorator(obj):
            self[name] = obj
            return obj

        return decorator
