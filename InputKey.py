from pynput.keyboard import Key, Listener 
shift = False

def show(key): 
    
    shift = False
    print('\nYou Entered {0}'.format( key)) 
    
    pressed_key = key
    if key == Key.shift:
        shift = True
    print(shift)
  
    if key == Key.delete: 
        # Stop listener 
        return False

with Listener(on_press = show) as listener:    
        listener.join() 