import random
import string

from PIL import Image

img_path_start_string = '<td><img src="'
img_path_end_string = '"></td>\n'

caption_start_strings = ['<tbody><tr><td> ', '<tr><td> ']
caption_end_string = '</td></tr>\n'

item_end_string = '</tbody></table></td>\n'

class ImageCaptionDataset:
    def __init__(
        self,
        data_dir=None,
    ):
        self._captions = {}
        self._img_ids = []
        self._data_dir = data_dir

    def get_captions(self, img_id):
        return self._captions[img_id]

    def get_image(self, img_id=None):
        if img_id is None:
            img_id = random.choice(self._img_ids)
        return {
            'id': img_id,
            'image': Image.open(self._data_dir + img_id),
        }

    def get_example(self, img_id=None):
        rd = self.get_image(img_id)
        return {
            'image': rd['image'], 
            'captions': self.get_captions(rd['id']),
        }

    def get_vocabulary(self):
        vocabulary = {}
        for img_id, captions in self._captions.items():
            for caption in captions:
                words = caption.split()
                for word in words:
                    word = word.lower().strip(',"')
                    try:
                        vocabulary[word] += 1
                    except KeyError:
                        vocabulary[word] = 1
        return vocabulary


class PASCAL(ImageCaptionDataset):
    def __init__(
        self,
        caption_filename='pascal-sentences.html',
        data_dir='./',
    ):
        super().__init__(data_dir)

        with open(caption_filename) as f:
            line_number = 0
            for line in f:
                line_number += 1
                
                if (
                    line[:len(img_path_start_string)] == img_path_start_string
                ):
                    img_path = line[len(img_path_start_string)
                                    :-len(img_path_end_string)]
                    self._captions[img_path] = []
                    self._img_ids.append(img_path)

                    while (line != item_end_string):
                        line = f.readline()
                        line_number += 1
                        for start_string in caption_start_strings:
                            if (line[:len(start_string)] == start_string):
                                line = line[len(start_string):]
                                caption = ''
                                while(line[-len(caption_end_string):]
                                      != caption_end_string):
                                    caption += line.strip()
                                    line = f.readline()
                                    line_number += 1
                                caption += line[:-len(caption_end_string)]
                                caption = caption.rstrip(
                                    string.whitespace + '.'
                                )
                                self._captions[img_path].append(caption)


class Flickr(ImageCaptionDataset):
    def __init__(
        self,
        caption_filename='',
        data_dir='./',
    ):
        super().__init__(data_dir)
        prev_img_id = None

        with open(caption_filename) as f:
            for line in f:
                filename_length = line.find('#')
                img_id = line[:filename_length]
                if img_id != prev_img_id:
                    self._captions[img_id] = []
                    self._img_ids.append(img_id)
                    prev_img_id = img_id

                raw_caption = line[filename_length + 3:]
                caption = raw_caption.rstrip(
                    string.whitespace + '.'
                )
                self._captions[img_id].append(caption)
            
