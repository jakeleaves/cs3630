import os
import zipfile


def main():
    # if file exists remove it
    if os.path.exists('submission.zip'):
        os.remove('submission.zip')

    zf = zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED)
    zf.write('brain.py')
    zf.write('world.py')
    zf.write('utils.py')
    zf.close()


if __name__ == "__main__":
    main()
