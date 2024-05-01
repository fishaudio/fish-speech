
function createGradioAnimation() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('__theme')) {
        params.set('__theme', 'light');
        window.location.search = params.toString();
    }

    var gradioApp = document.querySelector('gradio-app');
    if (gradioApp) {

        document.documentElement.style.setProperty('--my-200', '#80eeee');
        document.documentElement.style.setProperty('--my-50', '#ecfdf5');

        // gradioApp.style.position = 'relative';
        // gradioApp.style.backgroundSize = '200% 200%';
        // gradioApp.style.animation = 'moveJellyBackground 10s ease infinite';
        // gradioApp.style.backgroundImage = 'radial-gradient(circle at 0% 50%, var(--my-200), var(--my-50) 50%)';
        // gradioApp.style.display = 'flex';
        // gradioApp.style.justifyContent = 'flex-start';
        // gradioApp.style.flexWrap = 'nowrap';
        // gradioApp.style.overflowX = 'auto';

        // for (let i = 0; i < 6; i++) {
        //     var quan = document.createElement('div');
        //     quan.className = 'quan';
        //     gradioApp.insertBefore(quan, gradioApp.firstChild);
        //     quan.id = 'quan' + i.toString();
        //     quan.style.left = 'calc(var(--water-width) * ' + i.toString() + ')';
        //     var quanContainer = document.querySelector('.quan');
        //     if (quanContainer) {
        //         var shui = document.createElement('div');
        //         shui.className = 'shui';
        //         quanContainer.insertBefore(shui, quanContainer.firstChild)
        //     }
        // }
    }

    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontFamily = 'Maiandra GD, ui-monospace, monospace';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to Fish-Speech!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 200);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
