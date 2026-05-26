import functools

'''
Unit Conversions for ezfalcon.

Input/Output units (for user):
-----------------------------
- length: kpc
- velocity: km/s
- mass: Msun
- time: Gyr
- potential : (km/s)^2
- acceleration: km/s/Gyr
- angular_velocity: km/s/kpc
- angle: rad

Internal units (for simulation):
-----------------------------
- length: kpc
- velocity: kpc/Gyr
- mass: Msun
- time: Gyr
- potential : (kpc/Gyr)^2
- acceleration: kpc/Gyr^2
- angular_velocity: 1/Gyr
- angle: rad
'''

#: Gravitational constant [kpc^3 Msun^-1 Gyr^-2]
G_INTERNAL = 4.498502151469554e-06

#: G in mixed units [kpc (km/s)^2 Msun^-1] 
G_KPC_KMS = 4.300917270036279e-06

KM_TO_KPC = 3.2407792894443656e-17

#: 1 km/s in kpc/Gyr
KMS_TO_KPCGYR = 1.022712165045695

#: 1 kpc/Gyr in km/s  (approx 0.978)
KPCGYR_TO_KMS = 1.0 / KMS_TO_KPCGYR

#: 1 Gyr in Myr
GYR_TO_MYR = 1000.0

#: Legacy constants (kpc/Myr) — kept for backward compatibility
KMS_TO_KPCMYR = KMS_TO_KPCGYR / GYR_TO_MYR
KPCMYR_TO_KMS = 1.0 / KMS_TO_KPCMYR

INTERNAL_TO_USER_UNITS = {
    'length': 1.0,  # kpc
    'velocity': KPCGYR_TO_KMS,  # kpc/Gyr to km/s
    'mass': 1.0,  # Msun
    'time': 1.0,  # Gyr
    'energy': KPCGYR_TO_KMS**2,  # Msun (kpc/Gyr)^2 to Msun (km/s)^2
    'acceleration': KPCGYR_TO_KMS,  # kpc/Gyr^2 to km/s/Gyr
    'momentum': KPCGYR_TO_KMS,  # Msun kpc/Gyr to Msun km/s
    'angular_momentum': KPCGYR_TO_KMS,  # Msun kpc^2/Gyr to Msun kpc km/s
    'angular_velocity': KPCGYR_TO_KMS,  # 1/Gyr to km/s/kpc
    'angle': 1.0,  # rad
}

def unit_handler(unit_key):
    factor = INTERNAL_TO_USER_UNITS[unit_key]
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, return_internal=False, **kwargs):
            result = func(*args, **kwargs)
            if return_internal:
                return result
            return result * factor
        return wrapper
    return decorator
