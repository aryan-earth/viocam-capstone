class NoObjectsFoundException(Exception):
    def __init__(self):
        super().__init__('No objects found')

class NoVehiclesFoundException(Exception):
    def __init__(self):
        super().__init__('No vehicles found')

class NotWearingHelmetException(Exception):
    def __init__(self):
        super().__init__('Not wearing helmet')

class CantFindLicensePlateException(Exception):
    def __init__(self):
        super().__init__('Cannot find license plate')

class NoPersonFoundException(Exception):
    def __init__(self):
        super().__init__('No person found')