from rest_framework.schemas.openapi import AutoSchema

class HealthcareAPISchema(AutoSchema):
    """
    Custom schema class for the Healthcare API documentation.
    Provides better operation IDs and documentation.
    """
    
    def get_operation_id(self, path, method):
        """Generate better operation IDs for API endpoints."""
        method_name = method.lower()
        
        if hasattr(self.view, 'action'):
            model = self.view.__class__.__name__.replace('ViewSet', '')
            
            # Handle common ViewSet actions
            if self.view.action == 'list':
                return f"list{model}s"
            elif self.view.action == 'retrieve':
                return f"get{model}Detail"
            elif self.view.action == 'create':
                return f"create{model}"
            elif self.view.action == 'update':
                return f"update{model}"
            elif self.view.action == 'partial_update':
                return f"partial{model}Update" 
            elif self.view.action == 'destroy':
                return f"delete{model}"
            
            # Custom action handling
            return f"{self.view.action}{model}"
            
        return super().get_operation_id(path, method)
