'''
- Conv2d 3 > 32

###############################

- Conv2d 32 > 64 ; kernel=3, stride=2, padding=1
- Conv2d 64 > 32 ; kernel=1, stride=1, padding=1
- Conv2d 32 > 64 ; kernel=3, stride=1, padding=1
- RES (-3) & activation linear

- conv2d 64 > 128 ; kernel=3, stride=2, padding=1
- conv2d 128 > 64 ; kernel=1, stride=1, padding=1
- conv2d 64 > 128 ; kernel=3, stride=1, padding=1
- RES (-3) & activation linear

- conv2d 128 > 64 ; kernel=1, stride=1, padding=1
- conv2d 64 > 128 ; kernel=3, stride=1, padding=1
- RES (-3) & activation linear

###############################

- Conv2d 128 > 256 ; kernel=3, stride=2, padding=1
- Conv2d 256 > 128 ; kernel=1, stride=1, padding=1
- Conv2d 128 > 256 ; kernel=3, stride=1, padding=1
- RES (-3) & activation linear

- conv2d 256 > 128 ; kernel=1, stride=1, padding=1
- conv2d 128 > 256 ; kernel=3, stride=1, padding=1
- RES (-3) & activation linear

## Decrease channels = kernel 1 ##
## Increase channels = kernel 3 ##
'''