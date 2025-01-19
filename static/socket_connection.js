// socket_connection.js

// Establecer conexión persistente de SocketIO
var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

// Emitir evento al cerrar la pestaña
window.addEventListener('beforeunload', function (e) {
    socket.emit('client_closing_tab');
});


// Genera un ID único (puedes usar una librería o método más robusto)
const clientId = localStorage.getItem('clientId') || Date.now().toString();
localStorage.setItem('clientId', clientId);

// Conéctate al servidor con el ID del cliente
const socket = io.connect('http://localhost:5000', {
    query: { id: clientId }
});
