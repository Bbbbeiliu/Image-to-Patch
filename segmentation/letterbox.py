import cv2

def geometry(original_hw, size=(256, 256)):
    h, w = original_hw
    tw, th = (size, size) if isinstance(size, int) else size
    if min(h, w, tw, th) <= 0:
        raise ValueError('Image and target dimensions must be positive')
    scale = min(tw / w, th / h)
    nw, nh = (max(1, round(w * scale)), max(1, round(h * scale)))
    return ((th - nh) // 2, (tw - nw) // 2, nh, nw)

def letterbox(array, size=(256, 256), is_mask=False):
    tw, th = (size, size) if isinstance(size, int) else size
    top, left, nh, nw = geometry(array.shape[:2], size)
    resized = cv2.resize(array, (nw, nh), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(resized, top, th - nh - top, left, tw - nw - left, cv2.BORDER_CONSTANT, value=0)

def unpad(array, original_hw):
    top, left, nh, nw = geometry(original_hw, (array.shape[1], array.shape[0]))
    return array[top:top + nh, left:left + nw]

def restore(array, original_hw, is_mask=False):
    cropped = unpad(array, original_hw)
    return cv2.resize(cropped, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)

class LetterboxTransform:

    def __init__(self, augment=None, size=256):
        self.augment, self.size = (augment, size)

    def __call__(self, **data):
        if self.augment is not None:
            data = self.augment(**data)
        return {key: letterbox(value, self.size, is_mask=key != 'image') for key, value in data.items()}
