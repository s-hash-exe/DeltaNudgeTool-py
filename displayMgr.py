import matplotlib.pyplot as plt

class displayMgr():
    def __init__(s, mouseEvent, keyEvent):
        plt.set_cmap('gray')
        s.figNum = 1
        s.fig = plt.figure(1)
        #  Register callback functions for different mouse and keyboard events
        # Please note that it works in a separate thread (not necessary for conversion to javaScript)
        cid = s.fig.canvas.mpl_connect('button_press_event', mouseEvent)
        cid = s.fig.canvas.mpl_connect('button_release_event', mouseEvent)
        cid = s.fig.canvas.mpl_connect('motion_notify_event', mouseEvent)
        cid = s.fig.canvas.mpl_connect('key_press_event', keyEvent)
        return

    def clf(s):
        plt.clf()

    def pause(s, sec):
        plt.pause(sec)

    def close(s):
        plt.close(s.fig)

    def imshow(s, img, show=False, block=False):
        plt.figure(s.figNum)
        if img is not None:
            plt.imshow(img)
        if show:
            plt.show(block=block)

    def circle(s, cursorLoc, rad, fill=False, color='red'):
        circle = plt.Circle(cursorLoc, rad, color=color, fill=fill)
        axes = s.fig.gca()
        axes.add_patch(circle)
        return

    def text(s, xpos, ypos, line, fontsize=16, color='red'):
        plt.text(xpos, ypos, line, fontsize=fontsize, color='red')

    def plot(s, xPts, yPts, color='r'):
        plt.plot(xPts, yPts, color=color)